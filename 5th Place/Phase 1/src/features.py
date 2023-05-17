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
    """
    Extract some basic features such as name of airline, block time, day in week
    """
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


def get_diff_depart_vs_ts(test_pd, etd):
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
