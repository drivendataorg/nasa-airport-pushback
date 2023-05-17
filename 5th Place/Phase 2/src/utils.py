from collections import OrderedDict

import pandas as pd
import torch
from loguru import logger

from features import *

PUBLIC_DATA_PATH = "./data/raw/public/{}/{}_{}.csv.bz2"
LABEL_PATH = "./data/raw/phase2_train_labels_{}.csv.bz2"
PRIVATE_DATA_PATH = "./data/raw/private/{}/{}_{}_mfs.csv.bz2"

AIRPORTS = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]
AIRLINES = [
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
]


def load_public_airport_features(label_pd, airport_name):
    """Loads the public features for a set of flights from one airport.
    There are 5 categories of  public features which are:
    1. basic_feats: block_time (morning, afternoon,..) and day_in_week (Monday, Tuesday) all categorical features
    2. weather info: The weather information at the time of making prediction
    3. The difference between estimated departured time and the time of making prediction
    4. The number of flight winthin 1 hour before the time of making prediction
    5. Name of the airport
    """

    total_features = []
    etd_file_name = PUBLIC_DATA_PATH.format(airport_name, airport_name, "etd")
    etd = pd.read_csv(
        etd_file_name, parse_dates=["departure_runway_estimated_time", "timestamp"]
    )
    rw_file_name = PUBLIC_DATA_PATH.format(airport_name, airport_name, "runways")
    runways = pd.read_csv(rw_file_name, parse_dates=["timestamp"])

    weather_file_name = PUBLIC_DATA_PATH.format(airport_name, airport_name, "lamp")
    lamp = pd.read_csv(
        weather_file_name, parse_dates=["forecast_timestamp"]
    ).drop_duplicates(subset="forecast_timestamp")

    ###PUBLIC FEATURES
    #######################
    label_pd, features = get_diff_depart_vs_ts(label_pd, etd)  # extract_training data
    total_features += features
    label_pd, fetaures = get_weather_info(label_pd, lamp, airport_name)
    total_features += features
    label_pd, features = get_basic_feats(label_pd)
    total_features += features
    label_pd, features = get_num_dep_flights(label_pd, runways)
    total_features += features

    # We also add airport location as an additional cateogrical feature here
    for name in AIRPORTS:
        if name == airport_name:
            label_pd["airport_is_{}".format(name)] = 1
        else:
            label_pd["airport_is_{}".format(name)] = 0
        total_features += ["airport_is_{}".format(name)]

    return label_pd, total_features


def load_private_airport_airline_features(label_pd, airport_name, airline):
    mfs_file_name = PRIVATE_DATA_PATH.format(airport_name, airport_name, airline)
    mfs = pd.read_csv(mfs_file_name)
    mfs, features = get_meta_features(mfs)
    mfs = pd.merge(label_pd, mfs, on=["gufi"], how="left")

    return mfs, features


def extract_all_airlines_data():
    """
    Extract all labeled data for each airline
    The data for each airline is aggregated over 10 airports.

    """
    public_pd_list = []
    label_dict = {}
    for airport_name in AIRPORTS:
        logger.info(f"Extracting public data for {airport_name}")
        label_pd = pd.read_csv(
            LABEL_PATH.format(airport_name), parse_dates=["timestamp"]
        )
        label_pd["airline_name"] = label_pd["gufi"].str[:3]
        label_pd = label_pd[label_pd["airline_name"].isin(AIRLINES)]
        label_dict[airport_name] = label_pd
        pub_pd, pub_features = load_public_airport_features(label_pd, airport_name)
        public_pd_list.append(pub_pd)
    public_pd = pd.concat(public_pd_list)

    private_pd_list = []
    for airport_name in AIRPORTS:
        label_pd = label_dict[airport_name]
        for airline in AIRLINES:
            logger.info(f"Extracting private airline data for {airline}@{airport_name}")
            sub_label_pd = label_pd[label_pd["airline_name"] == airline][
                ["gufi", "timestamp"]
            ]
            priv_pd, priv_features = load_private_airport_airline_features(
                sub_label_pd, airport_name, airline
            )
            private_pd_list.append(priv_pd)

    private_pd = pd.concat(private_pd_list)
    # on timestamp
    pd_all = pd.merge(public_pd, private_pd, on=["gufi", "timestamp"], how="left")
    pd_all = pd_all.fillna(0)
    data_dict = {}
    data_dict["all_features"] = pub_features + priv_features
    for airline in AIRLINES:
        temp_pd = pd_all[pd_all["airline"] == airline]
        if len(temp_pd) > 0:
            data_dict[airline] = temp_pd

    return data_dict


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
