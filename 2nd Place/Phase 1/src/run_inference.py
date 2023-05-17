import numpy as np
import pandas as pd

from catboost import Pool, CatBoostRegressor
from typing import Any

import utils
from config import *


def load_model(solution_directory):
    print("Loading trained model from directory")
    airport_models = {}
    for airport in [
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
    ]:
        airport_models[airport] = CatBoostRegressor().load_model(
            solution_directory / f"catboost_{airport}_final"
        )

    airport_models["neg_val"] = CatBoostRegressor().load_model(
        solution_directory / f"catboost_neg_val"
    )
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

    features = features.merge(mfs, on="gufi")

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

    features = features.drop(
        columns=["gufi", "minutes_until_pushback", "airport"]
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

    features.drop(columns=["cloud_ceiling", "flight_type", "wind_speed"], inplace=True)

    for col in [
        "aircraft_engine_class",
        "major_carrier",
        "aircraft_type",
        "flight_number",
        "flight_arrival",
    ]:  # removed flight_type
        features[col] = features[col].astype(str)

    features = features[model[airport].feature_names_]

    features["predicted"] = model[airport].predict(features)
    # Re-process all negative prediction values
    features.loc[features["predicted"] < 0, "predicted"] = model["neg_val"].predict(
        features[features["predicted"] < 0].drop("predicted", axis=1)
    )

    partial_submission_format["minutes_until_pushback"] = features["predicted"]

    return partial_submission_format


if __name__ == "__main__":
    # Generate predictions for submission format using provided competition processing pipeline
    prediction_directory.mkdir(parents=True, exist_ok=True)
    utils.compute_predictions()
    utils.postprocess_predictions()
    print(f"Submission saved to {submission_path}")
