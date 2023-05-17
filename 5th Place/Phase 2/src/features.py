"""Predict 15 minutes before the Fuser estimated time of departure."""
import json
from pathlib import Path
from typing import Any

# from loguru import logger
import pandas as pd, numpy as np
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
    features = []
    test_pd["airline"] = test_pd["gufi"].apply(get_airline)
    test_pd["block_time"] = test_pd["timestamp"].apply(block_time).astype("category")
    test_pd["day_in_wk"] = test_pd["timestamp"].apply(day_in_wk).astype("category")
    for val in ["early_morning", "morning", "noon", "afternoon", "night"]:
        test_pd["block_time_in_{}".format(val)] = (
            test_pd["block_time"].apply(lambda x: x == val).astype(int)
        )
        features.append("block_time_in_{}".format(val))

    for val in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]:
        test_pd["day_is_{}".format(val)] = (
            test_pd["day_in_wk"].apply(lambda x: x == val).astype(int)
        )
        features.append("day_is_{}".format(val))

    return test_pd, features


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
        final_pd.sort_values(by="timestamp"),
        weather_pd.sort_values(by="timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("30h"),
        direction="backward",
    )
    features = [
        "wind_speed",
        "wind_direction",
        "temperature",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "precip",
    ]
    final_pd["precip"] = final_pd["precip"].apply(lambda x: x == True).astype(int)
    cloud_vals = ["OV", "CL", "FW", "BK", "SC"]
    for val in cloud_vals:
        final_pd["cloud_is_{}".format(val)] = (
            final_pd["cloud"].apply(lambda x: x == val).astype(int)
        )
        features += ["cloud_is_{}".format(val)]

    lightning_prob_vals = ["N", "L", "M", "H"]
    for val in lightning_prob_vals:
        final_pd["lightning_prob_is_{}".format(val)] = (
            final_pd["lightning_prob"].apply(lambda x: x == val).astype(int)
        )
        features += ["lightning_prob_is_{}".format(val)]

    return final_pd, features


def get_meta_features(mfs):
    """
    Extract meta features such as engine class
    """
    TOP_AIRCRAFT_TYPES = [
        "B738",
        "A321",
        "A320",
        "B739",
        "CRJ9",
        "B737",
        "A319",
        "E75L",
        "CRJ7",
        "CRJ2",
        "E170",
        "E145",
        "B752",
        "B763",
        "B712",
        "B38M",
        "A20N",
        "E75S",
        "A21N",
        "DH8D",
        "A306",
        "MD11",
        "B772",
        "B39M",
        "E190",
    ]

    TOP_MAJOR_CARRIERS = ["AAL", "UAL", "DAL", "JBU"]
    FLIGHT_TYPES = ["SCHEDULED_AIR_TRANSPORT", "GENERAL_AVIATION"]

    features = []

    for aircraft in TOP_AIRCRAFT_TYPES:
        mfs["is_aircraft_{}".format(aircraft)] = (
            mfs["aircraft_type"].apply(lambda x: x == aircraft).astype(int)
        )
        features.append("is_aircraft_{}".format(aircraft))

    for carrier in TOP_MAJOR_CARRIERS:
        mfs["is_carrier_{}".format(carrier)] = (
            mfs["major_carrier"].apply(lambda x: x == carrier).astype(int)
        )
        features.append("is_carrier_{}".format(carrier))

    for flight_type in FLIGHT_TYPES:
        mfs["is_flight_type_{}".format(flight_type)] = (
            mfs["flight_type"].apply(lambda x: x == flight_type).astype(int)
        )
        features.append("is_flight_type_{}".format(flight_type))

    final_cols = ["gufi"] + features

    return mfs[final_cols], features


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
    features = ["diff_depart_vs_ts"]

    return final_pd, features


def get_num_dep_flights(test_pd, runways):
    """
    EXTRACT numeric features that specify the number of flights within 1 hours before the
    time of making prediction
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
    features = ["num_flights"]
    return final_pd, features
