#
# Author: Kyler Robison
#
# Dependency for train_table script.
#

import math
from datetime import timedelta

import pandas as pd


def average_stand_time(
    origin_filtered: pd.DataFrame, standtimes_filtered: pd.DataFrame
) -> float:
    merged_df = pd.merge(origin_filtered, standtimes_filtered, on="gufi")

    merged_df["avg_stand_time"] = (
        merged_df["origin_time"] - merged_df["departure_stand_actual_time"]
    ).dt.total_seconds() / 60

    avg_stand_time: float = merged_df["avg_stand_time"].mean()
    if math.isnan(avg_stand_time):
        avg_stand_time = 0

    return round(avg_stand_time, 2)


def average_taxi_time(
    mfs: pd.DataFrame,
    standtimes: pd.DataFrame,
    runways_filtered: pd.DataFrame,
    departures: bool = True,
) -> float:
    mfs = mfs.loc[mfs["isdeparture"] == departures]

    merged_df = pd.merge(runways_filtered, mfs, on="gufi")
    merged_df = pd.merge(merged_df, standtimes, on="gufi")

    if departures:
        merged_df["taxi_time"] = (
            merged_df["departure_runway_actual_time"]
            - merged_df["departure_stand_actual_time"]
        ).dt.total_seconds() / 60
    else:
        merged_df["taxi_time"] = (
            merged_df["arrival_stand_actual_time"]
            - merged_df["arrival_runway_actual_time"]
        ).dt.total_seconds() / 60

    avg_taxi_time: float = merged_df["taxi_time"].mean()
    if math.isnan(avg_taxi_time):
        avg_taxi_time = 0

    return round(avg_taxi_time, 2)


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(
    df: pd.DataFrame, now: pd.Timestamp, hours: int
) -> pd.DataFrame:
    return df.loc[(df.timestamp > now - timedelta(hours=hours)) & (df.timestamp <= now)]
