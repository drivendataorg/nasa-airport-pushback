"""Predict 15 minutes before the Fuser estimated time of departure."""
import json
from pathlib import Path
from typing import Any

# from loguru import logger
import pandas as pd, numpy as np
import xgboost

import datetime
from datetime import timedelta


def get_airline(x):
    """
    Extract airline name e.g., AAL (American Airlines)
    """
    name_number = x.split(".")[0]
    name = "".join([x for x in name_number if not x.isnumeric()])

    return name


def day_in_wk(date_time):
    """
    Extract day in week based on date

    """
    date_time = str(date_time)
    dates = date_time.split(" ")[0]
    year, month, day = (int(x) for x in dates.split("-"))
    ans = datetime.date(year, month, day)

    return ans.strftime("%A")


def block_time(x):
    """
    Get block time of the day (early morning, morning, noon, afternoon and night) based on hour
    """
    x = str(x)
    x = x.split(" ")[1]
    h = int(x.split(":")[0])
    if h <= 6:
        return "early morning"
    elif h <= 10:
        return "morning"
    elif h <= 14:
        return "noon"
    elif h <= 18:
        return "afternoon"
    else:
        return "night"


def get_basic_feats(test_pd):
    cat_features = ["airline", "block_time", "day_in_wk"]
    test_pd["airline"] = test_pd["gufi"].apply(get_airline).astype("category")
    test_pd["block_time"] = test_pd["timestamp"].apply(block_time).astype("category")
    test_pd["day_in_wk"] = test_pd["timestamp"].apply(day_in_wk).astype("category")
    for cat in cat_features:
        if cat in test_pd.columns.tolist():
            test_pd[cat] = test_pd[cat].astype("category")

    feat_dict = {"numeric": [], "cat": []}

    return test_pd, feat_dict


###################


def get_weather_info(test_pd, lamp, airport_name):
    """
    test_pd: will be either label data/submission data
    and should have timestamp and airportname column
    aiport_name:  name of the airport to extract weather information
    """

    weather_pd = lamp.copy()
    weather_pd["timestamp"] = weather_pd["forecast_timestamp"]
    # We only keep the date the prediction is about (e.g., tomorrow)
    # So only forecast_timestamp will be kept.

    final_pd = test_pd[test_pd["airport"] == airport_name]
    final_pd = pd.merge_asof(
        test_pd.sort_values(by="timestamp"),
        weather_pd.sort_values(by="timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("30h"),
        direction="backward",
    )

    def convert_precip(x):
        if x == True:
            return "1"
        else:
            return "0"

    final_pd["precip"] = final_pd["precip"].apply(convert_precip).astype("category")
    cat_features = ["precip", "cloud", "lightning_prob"]
    numeric_features = [
        "wind_speed",
        "wind_direction",
        "temperature",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
    ]

    feat_dict = {"numeric": numeric_features, "cat": cat_features}

    return final_pd, feat_dict


###########


def get_meta_features(test_pd, mfs):
    """
    Extract meta features such as engine class
    """
    meta_cat_cols = [
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
    ]
    final_pd = pd.merge(test_pd, mfs, on="gufi", how="left")

    feat_dict = {"numeric": [], "cat": meta_cat_cols}
    for col in meta_cat_cols:
        final_pd[col] = final_pd[col].astype("category")

    return final_pd, feat_dict


def get_diff_depart_vs_ts(test_pd, etd=None):
    """
    Get the difference between estimated departutre time and the timestamp in test_pd
    test_pd can be either submission or pushback label data
    In the banchmark diff_depart_vs_ts - 15 is the estimation of minutes_until_pushback
    """

    etd = etd.copy()
    final_pd = pd.merge_asof(
        test_pd.sort_values(by=["timestamp"]),
        etd.sort_values(by=["timestamp"]),
        on="timestamp",
        by="gufi",
        tolerance=pd.Timedelta("30h"),
        direction="backward",
    )

    final_pd["diff_depart_vs_ts"] = pd.to_timedelta(
        final_pd["departure_runway_estimated_time"] - final_pd["timestamp"], unit="m"
    )
    final_pd["diff_depart_vs_ts"] = final_pd["diff_depart_vs_ts"].astype(
        "timedelta64[m]"
    )
    final_pd.fillna(value={"diff_depart_vs_ts": 0}, inplace=True)
    final_pd["diff_depart_vs_ts"] = final_pd["diff_depart_vs_ts"].astype(int)
    feat_dict = {"numeric": ["diff_depart_vs_ts"], "cat": []}

    return final_pd, feat_dict


def get_dep_rw(test_pd, runways):
    """
    EXTRACT categorical features that specify the runway gate of a departure flight e.g., 18L
    NOTE THAT all timestamp in the runways is always less than timestamp in submission and pushback label data

    """
    rw = runways.copy()
    rw = rw.drop_duplicates(subset=["gufi"])  #
    final_pd = pd.merge_asof(
        test_pd.sort_values(by="timestamp"),
        rw.sort_values(by="timestamp"),
        on="timestamp",
        by="gufi",
        tolerance=pd.Timedelta("30h"),
        direction="backward",
    )

    final_pd["departure_runway_actual"].fillna("NA", inplace=True)
    final_pd["departure_runway_actual"] = final_pd["departure_runway_actual"].astype(
        "category"
    )

    feat_dict = {"numeric": [], "cat": ["departure_runway_actual"]}

    return final_pd, feat_dict


def get_num_dep_flights(test_pd, runways):
    """
    EXTRACT categorical features that specify the runway gate of a departure flight e.g., 18L
    NOTE THAT all timestamp in the runways is always less than timestamp in submission and pushback label data

    """

    rw = runways.copy()
    rw = rw.drop_duplicates(subset=["gufi"])
    rw["prev_timestamp_1hr"] = rw["timestamp"].apply(lambda x: x - timedelta(hours=1))
    rw["prev_date_hour"] = rw["prev_timestamp_1hr"].apply(
        lambda x: str(x).split(":")[0]
    )

    pd01 = rw.groupby(["prev_date_hour"]).agg({"gufi": "nunique"}).reset_index()
    pd01.columns = ["date_hour", "num_flights"]
    test_pd["date_hour"] = test_pd["timestamp"].apply(lambda x: str(x).split(":")[0])
    final_pd = pd.merge(test_pd, pd01, on="date_hour", how="left")

    feat_dict = {"numeric": ["num_flights"], "cat": []}

    return final_pd, feat_dict


##############
# TODO adding features computing func

###################
# PREDICT PATH

airports_list = [
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


def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    model = {}
    lr = 2.5
    for airport_name in airports_list:
        model[airport_name] = xgboost.XGBRegressor()
        model[airport_name].load_model(
            solution_directory / "v5_model_{}_{}.json".format(airport_name, lr)
        )
    return model


def predict(
    config: pd.DataFrame = None,
    etd: pd.DataFrame = None,
    first_position: pd.DataFrame = None,
    lamp: pd.DataFrame = None,
    mfs: pd.DataFrame = None,
    runways: pd.DataFrame = None,
    standtimes: pd.DataFrame = None,
    tbfm: pd.DataFrame = None,
    tfm: pd.DataFrame = None,
    airport: str = None,
    prediction_time: pd.Timestamp = None,
    partial_submission_format: pd.DataFrame = None,
    model: Any = None,
    solution_directory: Path = None,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time."""
    # logger.debug("First submisstion of Cường's team")
    if airport not in model:
        airport_name = "KATL"
        airport_model = model["KATL"]
    else:
        airport_name = airport
        airport_model = model[airport]
    features = []
    cat_features = []
    USED_COLS = [
        x
        for x in partial_submission_format.columns.tolist()
        if x not in ["minutes_until_pushback"]
    ]
    submission_pd = partial_submission_format.copy()
    # derparture etd features
    submission_pd, fts_dict = get_diff_depart_vs_ts(submission_pd, etd=etd)
    features += fts_dict["numeric"] + fts_dict["cat"]
    cat_features += fts_dict["cat"]
    # lamp features
    submission_pd, fts_dict = get_weather_info(submission_pd, lamp, airport)
    features += fts_dict["numeric"] + fts_dict["cat"]
    cat_features += fts_dict["cat"]
    # basic features
    submission_pd, fts_dict = get_basic_feats(submission_pd)
    features += fts_dict["numeric"] + fts_dict["cat"]
    cat_features += fts_dict["cat"]
    # TODO adding new features
    submission_pd, meta_feat_dict = get_meta_features(submission_pd, mfs)
    features += meta_feat_dict["numeric"] + meta_feat_dict["cat"]
    cat_features += meta_feat_dict["cat"]

    submission_pd, rw_feat_dict = get_dep_rw(submission_pd, runways)
    features += rw_feat_dict["numeric"] + rw_feat_dict["cat"]
    cat_features += rw_feat_dict["cat"]
    submission_pd, dep_feat_dict = get_num_dep_flights(submission_pd, runways)
    features += dep_feat_dict["numeric"] + dep_feat_dict["cat"]
    cat_features += dep_feat_dict["cat"]

    # categorize cat features
    values = {v: -1 for v in features if v not in cat_features}
    for v in cat_features:
        submission_pd[v] = submission_pd[v].astype("string")
        submission_pd[v].fillna("NA", inplace=True)
        submission_pd[v] = submission_pd[v].astype("category")
    submission_pd = submission_pd.fillna(values)
    submission_pd = pd.get_dummies(submission_pd, columns=cat_features)
    old_features = model[airport_name].get_booster().feature_names
    for x in old_features:
        if x not in submission_pd.columns.tolist():
            submission_pd[x] = 0
    # predict
    y_pred = airport_model.predict(submission_pd[old_features])
    submission_pd["minutes_until_pushback"] = y_pred
    prediction = partial_submission_format.copy()
    prediction = pd.merge(
        prediction[USED_COLS],
        submission_pd[["gufi", "timestamp", "minutes_until_pushback"]],
        on=["gufi", "timestamp"],
        how="left",
    )

    # submission_pd['minutes_until_pushback'] = y_pred
    prediction["minutes_until_pushback"] = prediction["minutes_until_pushback"].apply(
        lambda x: max(1, int(x))
    )
    prediction.fillna(value={"minutes_until_pushback": 1}, inplace=True)
    prediction = prediction[partial_submission_format.columns.tolist()]
    prediction = prediction.astype(partial_submission_format.dtypes.to_dict())
    # create final prediction

    # prediction["minutes_until_pushback"] = prediction["minutes_until_pushback"] .apply(lambda x: max(int(x),1))

    return prediction
