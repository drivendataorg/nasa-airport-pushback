from utils import *
from network import *

USED_COLS = ["gufi", "timestamp", "minutes_until_pushback"]

if __name__ == "__main__":
    # load global model
    params = np.load("./models/federated_weights.npz")
    net = MLP(56, 4)
    state_dict = {}
    for (name, w), z in zip(net.named_parameters(), list(params.keys())):
        state_dict[name] = torch.Tensor(params[z])
    net.load_state_dict(state_dict, strict=True)

    submission_format = pd.read_csv(
        data_directory / "raw" / "submission_format.csv",
        parse_dates=["timestamp"],
        compression="bz2",
    )
    submission_format["airline_name"] = submission_format["gufi"].str[:3]
    public_pd_list = []
    for airport_name in AIRPORTS:
        pub_pd, pub_features = load_public_airport_features(
            submission_format, airport_name
        )
        public_pd_list.append(pub_pd)
    public_pd = pd.concat(public_pd_list)

    private_pd_list = []
    for airport_name in AIRPORTS:
        for airline in AIRLINES:
            partial_submission_format = submission_format[
                (submission_format["airline_name"] == airline)
                & (submission_format["airport"] == airport_name)
            ]
            priv_pd, priv_features = load_private_airport_airline_features(
                partial_submission_format, airport_name, airline
            )
            private_pd_list.append(priv_pd)

    private_pd = pd.concat(private_pd_list)
    pd_all = pd.merge(
        public_pd,
        private_pd,
        on=["gufi", "timestamp", "airport", "airline_name"],
        how="left",
    )
    pd_all = pd_all.fillna(0)
    temp_pd = pd.merge(
        submission_format,
        pd_all,
        on=["gufi", "timestamp", "airport", "airline_name"],
        how="left",
    ).fillna(0)
    features = pub_features + priv_features
    temp_pd["minutes_until_pushback"] = (
        net(torch.Tensor(temp_pd[features].values)).detach().numpy()
    )
    temp_pd["minutes_until_pushback"] = temp_pd["minutes_until_pushback"].clip(
        upper=300
    )
    submission_format_final = pd.merge(
        submission_format[["gufi", "timestamp"]],
        temp_pd[USED_COLS],
        on=["gufi", "timestamp"],
        how="left",
    )
    submission_format_final.to_csv("./processed/submission.csv", index=False)
