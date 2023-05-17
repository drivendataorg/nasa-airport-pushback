import os
import sys

sys.path.append(os.getcwd())

from src import *

import utils
import glob
from client import CNN, tree_predictions
from config import airports


def load_model(solution_directory):
    print("Loading trained model from directory")
    airport_models = {}
    for airport in airports:
        tree_directory = os.getcwd() + f"/trees/{airport}/"
        weights_path = os.getcwd() + f"/model/{airport}/round-5-weights.npy"
        num_clients = len(glob.glob(tree_directory + "tree-*"))
        model = CNN(num_clients, client_tree_num)
        weights = np.load(weights_path, allow_pickle=True)
        model.set_weights(weights)
        model.to(device)
        model = model.double()

        trees = []
        for i in range(num_clients):
            tree_path = tree_directory + f"/tree-{i}"
            tree = CatBoostRegressor().load_model(tree_path)
            trees.append(tree)
        airport_models[airport] = (model, trees)

    return airport_models


def predict(
    config: pd.DataFrame,
    etd: pd.DataFrame,
    first_position: pd.DataFrame,
    lamp: pd.DataFrame,
    mfs: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tbfm: pd.DataFrame,
    tfm: pd.DataFrame,
    airport: str,
    prediction_time: pd.Timestamp,
    partial_submission_format: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time.

    Returns:
        pd.DataFrame: Predictions for all of the flights in the partial_submission_format
    """
    if len(partial_submission_format) == 0:
        return partial_submission_format

    etd = etd.sort_values(["gufi", "timestamp"]).reset_index(drop=True)
    etd["hg"] = etd["gufi"].shift(1)
    etd["hts"] = etd["timestamp"].shift(1)
    etd["hd"] = etd["departure_runway_estimated_time"].shift(1)
    etd.loc[etd.gufi != etd.hg, ["hts", "hd"]] = np.nan

    etd["ts_diff"] = (etd["timestamp"] - etd["hts"]).dt.total_seconds()
    etd["etd_diff"] = (
        etd["departure_runway_estimated_time"] - etd["hd"]
    ).dt.total_seconds()
    etd["etd_slope"] = etd["etd_diff"] / etd["ts_diff"]
    etd = etd[(etd.etd_slope != np.inf) & (etd.etd_slope != -np.inf)]

    etd["ts_diff_cumsum"] = etd.groupby(["gufi"])["ts_diff"].cumsum()
    etd["etd_diff_cumsum"] = etd.groupby(["gufi"])["etd_diff"].cumsum()
    etd["etd_slope_cumsum"] = etd["etd_diff_cumsum"] / etd["ts_diff_cumsum"]
    etd.drop(
        columns=[
            "ts_diff",
            "etd_diff",
            "hg",
            "hts",
            "hd",
            "ts_diff_cumsum",
            "etd_diff_cumsum",
        ],
        inplace=True,
    )

    etd = etd.sort_values("timestamp")
    etd.insert(loc=2, column="etd_timestamp", value=etd["timestamp"])
    features = pd.merge_asof(
        partial_submission_format, etd, on="timestamp", by="gufi", direction="backward"
    )

    if len(lamp) > 0:
        nearest_time = min(
            lamp.forecast_timestamp.unique(), key=lambda sub: abs(sub - prediction_time)
        )
        latest_lamp = lamp[lamp.forecast_timestamp == nearest_time]
        latest_lamp = latest_lamp[latest_lamp.timestamp == latest_lamp.timestamp.max()]
        latest_lamp.drop(columns=["timestamp", "forecast_timestamp"], inplace=True)
        features = features.assign(**latest_lamp.iloc[0])
    else:
        for col in lamp.columns:
            if col != "timestamp" and col != "forecast_timestamp":
                features[col] = np.NaN
                print("No lamp values: Use Nan instead")

    features["flight_number"] = features.gufi.apply(lambda x: str(x).split(".")[0])
    features["flight_arrival"] = features.gufi.apply(lambda x: str(x).split(".")[2])

    features = features.drop(columns=["airport"])  # airports are inferenced separately

    features["departure_runway_estimated_time"] = (
        features["departure_runway_estimated_time"] - features["timestamp"]
    ).dt.total_seconds() / 60.0
    features.rename(
        {"departure_runway_estimated_time": "minutes_until_departure"},
        axis="columns",
        inplace=True,
    )

    features["etd_timestamp"] = (
        features["timestamp"] - features["etd_timestamp"]
    ).dt.total_seconds() / 60.0
    features.rename(
        {"etd_timestamp": "minutes_after_etd"}, axis="columns", inplace=True
    )

    features.insert(
        loc=2,
        column="hour_cos",
        value=np.cos(features.timestamp.dt.hour * (2.0 * np.pi / 24)),
    )
    features.insert(
        loc=2,
        column="hour_sin",
        value=np.sin(features.timestamp.dt.hour * (2.0 * np.pi / 24)),
    )
    features.insert(
        loc=2,
        column="dow_cos",
        value=np.cos(features.timestamp.dt.day_of_week * (2.0 * np.pi / 7)),
    )
    features.insert(
        loc=2,
        column="dow_sin",
        value=np.sin(features.timestamp.dt.day_of_week * (2.0 * np.pi / 7)),
    )
    features.insert(
        loc=2,
        column="day_cos",
        value=np.cos(features.timestamp.dt.day * (2.0 * np.pi / 365)),
    )
    features.insert(
        loc=2,
        column="day_sin",
        value=np.sin(features.timestamp.dt.day * (2.0 * np.pi / 365)),
    )
    features.insert(
        loc=2,
        column="month_cos",
        value=np.cos(features.timestamp.dt.month * (2.0 * np.pi / 12)),
    )
    features.insert(
        loc=2,
        column="month_sin",
        value=np.sin(features.timestamp.dt.month * (2.0 * np.pi / 12)),
    )
    features.insert(loc=2, column="year", value=features.timestamp.dt.year)
    features.drop("timestamp", axis=1, inplace=True)

    # Feature encoding
    features["precip"] = features["precip"].map({True: 1, False: 0})
    features["lightning_prob"] = features["lightning_prob"].map(
        {"N": 0, "L": 1, "M": 2, "H": 3}
    )
    features["cloud"] = features["cloud"].map(
        {"CL": 0, "FW": 1, "SC": 2, "BK": 3, "OV": 4}
    )

    features["flight_number"] = features["flight_number"].str.extract("(\d+)")
    features.fillna(0, inplace=True)
    features["flight_number"] = features["flight_number"].astype(int)

    features.drop(
        columns=["cloud_ceiling", "wind_speed", "flight_arrival"], inplace=True
    )

    features = features.merge(mfs, on="gufi")

    feature_mask = {
        "KATL": [
            "A321",
            "B712",
            "B739",
            "CRJ9",
            "B752",
            "A320",
            "B738",
            "B737",
            "CRJ2",
            "Other",
        ],
        "KCLT": [
            "CRJ9",
            "A321",
            "B738",
            "CRJ7",
            "A319",
            "E145",
            "A320",
            "E75S",
            "E75L",
            "Other",
        ],
        "KDEN": [
            "B738",
            "B737",
            "CRJ2",
            "E75L",
            "A320",
            "B739",
            "A319",
            "A321",
            "A20N",
            "Other",
        ],
        "KDFW": [
            "B738",
            "A321",
            "E170",
            "A319",
            "CRJ7",
            "CRJ9",
            "E145",
            "A320",
            "E75L",
            "Other",
        ],
        "KJFK": [
            "A320",
            "A321",
            "CRJ9",
            "B738",
            "E75S",
            "E190",
            "B739",
            "A21N",
            "B763",
            "Other",
        ],
        "KMEM": [
            "B763",
            "A306",
            "MD11",
            "B752",
            "B77L",
            "DC10",
            "CRJ9",
            "B738",
            "E75L",
            "Other",
        ],
        "KMIA": [
            "B738",
            "B38M",
            "A321",
            "E170",
            "A319",
            "A320",
            "B737",
            "B772",
            "B763",
            "Other",
        ],
        "KORD": [
            "CRJ2",
            "B738",
            "CRJ7",
            "E75L",
            "E145",
            "E170",
            "B739",
            "A320",
            "A319",
            "Other",
        ],
        "KPHX": [
            "B737",
            "B738",
            "A321",
            "CRJ7",
            "CRJ9",
            "A320",
            "A319",
            "B38M",
            "A21N",
            "Other",
        ],
        "KSEA": [
            "B739",
            "E75L",
            "B738",
            "DH8D",
            "B737",
            "A320",
            "B39M",
            "A321",
            "BCS1",
            "Other",
        ],
    }

    features["aircraft_type"] = features["aircraft_type"].apply(
        lambda x: x if x in feature_mask[airport] else "Other"
    )
    features["isdeparture"] = features["isdeparture"].map({True: 1, False: 0})

    features["aircraft_type"] = features["aircraft_type"].astype("category")
    features["aircraft_type"] = features["aircraft_type"].cat.codes

    features.drop(
        columns=["gufi", "major_carrier", "flight_type", "aircraft_engine_class"],
        inplace=True,
    )

    all_preds = np.array([])
    net, trees = model[airport]

    for tree in trees:
        tree_preds = []
        for pred in tree_predictions(tree, features.to_numpy()):
            tree_preds.append([pred])

        tree_preds = np.transpose(tree_preds)
        if len(all_preds) == 0:
            all_preds = tree_preds
        else:
            all_preds = np.append(all_preds, tree_preds, axis=2)

    all_preds = (
        torch.from_numpy(all_preds).double().to(device)
    )  # convert to pytorch tensor
    all_preds = net(all_preds).squeeze()

    partial_submission_format["minutes_until_pushback"] = (
        all_preds.cpu().detach().numpy()
    )

    return partial_submission_format


if __name__ == "__main__":
    # Generate predictions for submission format using provided competition processing pipeline
    prediction_directory.mkdir(parents=True, exist_ok=True)
    utils.compute_predictions()
    utils.postprocess_predictions()
    print(f"Submission saved to {submission_path}")
