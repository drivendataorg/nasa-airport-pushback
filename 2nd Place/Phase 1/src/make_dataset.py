import numpy as np
import pandas as pd

from catboost import Pool, CatBoostRegressor
from datetime import datetime, timedelta

from utils import _load_features
from pathlib import Path
from config import *


# Runway feature ended up not being used in the final model
def merge_runway_features(full_df, runways):
    departures = runways[runways.departure_runway_actual_time.isna() == False][
        ["gufi", "departure_runway_actual_time"]
    ].sort_values(by=["departure_runway_actual_time"])
    departures["dep20_time"] = (
        departures.departure_runway_actual_time
        - departures.departure_runway_actual_time.shift(20)
    ) / timedelta(minutes=1)
    full_df = pd.merge_asof(
        full_df.sort_values("timestamp"),
        departures[["dep20_time", "departure_runway_actual_time"]].rename(
            columns={"departure_runway_actual_time": "timestamp"}
        ),
        on="timestamp",
        direction="backward",
    )

    return full_df


# Queue feature ended up not being used in final model
def get_queue_feature(etd, full_df):
    print("Extract airplane etd queue")

    # create a copy of etd and drop estimates made after departure (to save memory and speed operations)
    etdh = etd[etd["departure_runway_estimated_time"] > etd["timestamp"]].copy()

    # split data into 15 minutes intervals
    etdh["etd_time_group"] = etdh.departure_runway_estimated_time + (
        datetime(2020, 1, 1, 0, 0, 0, 0) - etdh.departure_runway_estimated_time
    ) % timedelta(minutes=15)
    etdh["data_time_group"] = etdh.timestamp + (
        datetime(2020, 1, 1, 0, 0, 0, 0) - etdh.timestamp
    ) % timedelta(minutes=15)
    etdh = etdh[etdh["etd_time_group"] > etdh["data_time_group"]]

    # in each 15 minutes interval, drop all estimates except for the most recent
    etdh = etdh.sort_values(by=["timestamp"])
    etdh = etdh.drop_duplicates(subset=["gufi", "data_time_group"], keep="last")

    df1 = etdh[["gufi", "etd_time_group"]].drop_duplicates(
        subset=["gufi", "etd_time_group"]
    )
    df1["data_time_group"] = df1.etd_time_group - timedelta(minutes=15)
    etdh_x = pd.merge_asof(
        df1[["gufi", "data_time_group"]].sort_values("data_time_group"),
        etdh[
            ["data_time_group", "gufi", "departure_runway_estimated_time"]
        ].sort_values("data_time_group"),
        on="data_time_group",
        by="gufi",
        direction="backward",
    )

    for n in range(2, 23):
        df1["data_time_group"] = df1.etd_time_group - n * timedelta(minutes=15)
        etdh_x1 = pd.merge_asof(
            df1.sort_values("data_time_group"),
            etdh[
                ["data_time_group", "gufi", "departure_runway_estimated_time"]
            ].sort_values("data_time_group"),
            on="data_time_group",
            by="gufi",
            direction="backward",
        )
        etdh_x = pd.concat([etdh_x, etdh_x1])
        etdh_x = etdh_x.drop_duplicates(subset=["gufi", "data_time_group"])

    etdh_x = etdh_x[etdh_x.departure_runway_estimated_time.isna() == False]
    etdh_x["queue"] = etdh_x.groupby("data_time_group")[
        "departure_runway_estimated_time"
    ].rank()

    full_df["data_time_group"] = full_df.timestamp + (
        datetime(2020, 1, 1, 0, 0, 0, 0) - full_df.timestamp
    ) % timedelta(minutes=15)
    full_df = full_df.merge(
        etdh_x[["queue", "gufi", "data_time_group"]],
        how="left",
        on=["gufi", "data_time_group"],
    )
    full_df = pd.merge_asof(
        full_df.sort_values("data_time_group"),
        etdh[["data_time_group", "gufi"]].sort_values("data_time_group"),
        on="data_time_group",
        by="gufi",
        direction="backward",
    )
    full_df.drop("data_time_group", axis=1, inplace=True)


def merge_etd_features(labels, etd):
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

    etd = etd.sort_values("timestamp").reset_index(drop=True)
    etd.insert(loc=2, column="etd_timestamp", value=etd["timestamp"])
    full_df = pd.merge_asof(
        labels, etd, on="timestamp", by="gufi", direction="backward"
    )

    return full_df


def get_lamp_features(lamp):
    lamp = lamp.sort_values("timestamp").reset_index(drop=True)
    new_lamp = lamp.copy()
    for i in range(1, 4):
        temp = lamp.copy()
        temp["forecast_timestamp"] = temp["forecast_timestamp"] + pd.Timedelta(
            minutes=15 * i
        )
        new_lamp = pd.concat([new_lamp, temp])
    new_lamp = new_lamp.drop_duplicates()
    new_lamp = new_lamp.sort_values(["timestamp", "forecast_timestamp"]).reset_index(
        drop=True
    )

    return new_lamp


# Features are extracted such that future data is never used
def extract_features(airport, save=False):
    print("Loading training label file")
    train_labels = pd.read_csv(
        data_directory / f"prescreened_train_labels_{airport}.csv",
        parse_dates=["timestamp"],
    )
    train_labels = train_labels.sort_values("timestamp").reset_index(drop=True)

    print("Loading feature files")
    for feature_name in ["etd", "lamp", "mfs", "runways"]:
        _load_features(airport, feature_name)
    etd = FEATURES[airport]["etd"]
    lamp = FEATURES[airport]["lamp"]
    mfs = FEATURES[airport]["mfs"]
    runways = FEATURES[airport]["runways"]

    print("Processing etd features")
    full_df = merge_etd_features(train_labels, etd)

    print("Processing mfs features")
    full_df = full_df.merge(mfs, on="gufi")

    print("Processing lamp features")
    lamp = get_lamp_features(lamp)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df["forecast_timestamp"] = full_df["timestamp"]
    full_df = pd.merge_asof(
        full_df, lamp, on="timestamp", by="forecast_timestamp", direction="backward"
    )
    full_df.drop("forecast_timestamp", axis=1, inplace=True)

    print("Extract gufi features")
    full_df = full_df.sort_values(["timestamp", "gufi"]).reset_index(drop=True)
    full_df["flight_number"] = full_df.gufi.apply(lambda x: str(x).split(".")[0])
    full_df["flight_arrival"] = full_df.gufi.apply(lambda x: str(x).split(".")[2])

    # Save extracted features as zipped pickle file
    if save:
        print("Saving file")
        full_df.to_pickle(train_directory / f"{airport}_full.pkl.zip")

    return full_df


def encode_features(features, airport, save=False):
    features = features.drop(
        columns=["gufi", "airport"]
    )  # airports are inferenced separately

    # Reduce aircraft_type cardinality
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
            "A319",
            "B763",
            "E75S",
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
            "CRJ7",
            "B38M",
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
            "A21N",
            "B789",
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
            "B752",
            "B772",
            "A319",
            "B764",
            "E75L",
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
            "B737",
            "A321",
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
            "B788",
            "A21N",
            "B752",
            "E75L",
            "B739",
            "B77W",
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
            "A321",
            "B737",
            "E75S",
            "B753",
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
            "B739",
            "E75L",
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
    features["isdeparture"] = features["isdeparture"].map({True: 1, False: 0})
    features["precip"] = features["precip"].map({True: 1, False: 0})
    features["lightning_prob"] = features["lightning_prob"].map(
        {"N": 0, "L": 1, "M": 2, "H": 3}
    )
    features["cloud"] = features["cloud"].map(
        {"CL": 0, "FW": 1, "SC": 2, "BK": 3, "OV": 4}
    )

    for col in [
        "aircraft_engine_class",
        "major_carrier",
        "flight_type",
        "aircraft_type",
        "flight_number",
        "flight_arrival",
    ]:  # removed airport as feature
        features[col] = features[col].astype(str)

    # Drop unimportant features
    features.drop(columns=["cloud_ceiling", "flight_type", "wind_speed"], inplace=True)

    # Save encoded features as zipped pickle file
    if save:
        features.to_pickle(train_directory / f"{airport}_features_with_null.pkl.zip")

    return features


def get_negative_predictions():
    neg_pred = []
    for airport in airports:
        data_df = pd.read_pickle(
            train_directory / f"{airport}_features_with_null.pkl.zip"
        )

        X = data_df.drop(["minutes_until_pushback"], axis=1)
        y = data_df["minutes_until_pushback"]

        cat_cols = [
            "aircraft_engine_class",
            "major_carrier",
            "aircraft_type",
            "flight_number",
            "flight_arrival",
        ]  # removed flight_type
        for col in cat_cols:
            X[col] = X[col].astype(str)

        model = CatBoostRegressor().load_model(
            model_directory / f"catboost_{airport}_final"
        )
        y_pred = model.predict(X)

        data_df.insert(loc=0, column="predicted", value=y_pred)
        neg_pred.append(data_df[data_df.predicted < 0])

    neg_pred = pd.concat(neg_pred, copy=False)

    return neg_pred


def make_dataset(airport, save=True):
    print(f"Extracting features for {airport}")
    train_directory.mkdir(parents=True, exist_ok=True)
    features = extract_features(airport, save=False)
    data_df = encode_features(features, airport, save=save)

    return data_df
