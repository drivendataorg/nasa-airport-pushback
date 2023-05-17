#
# Authors:
# - Kyler Robison
# - Yudong Lin
#
# Dependency for train_table script.
#

import math
from datetime import timedelta

import pandas as pd


def get_average_difference_of_columns(_df: pd.DataFrame, col1: str, col2: str, round_to: int = 2):
    _df["difference_temp_data"] = (_df[col1] - _df[col2]).dt.total_seconds() / 60
    avg_delay: float = _df["difference_temp_data"].mean()  # type: ignore
    return round(0 if math.isnan(avg_delay) else avg_delay, round_to)


def get_average_difference(df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str, round_to: int = 2) -> float:
    return get_average_difference_of_columns(df1.merge(df2, on="gufi"), col1, col2, round_to)


def average_departure_delay(
    etd_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "departure_runway_actual_time"
) -> float:
    return get_average_difference(etd_filtered, runways_filtered, column_name, "departure_runway_estimated_time")


def average_arrival_delay(
    tfm_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "arrival_runway_actual_time"
) -> float:
    """
    Difference between the time that the airplane was scheduled to arrive and the time it is
    truly arriving
    """
    return get_average_difference(tfm_filtered, runways_filtered, column_name, "arrival_runway_estimated_time")


def average_arrival_delay_on_prediction(
    tfm_filtered: pd.DataFrame, tbfm_filtered: pd.DataFrame, column_name: str = "arrival_runway_estimated_time"
) -> float:
    """
    Difference between the time that the airplane was scheduled to arrive and the time it is currently
    estimated to arrive
    """
    return get_average_difference(tfm_filtered, tbfm_filtered, column_name, "scheduled_runway_estimated_time")


def average_stand_time(origin_filtered: pd.DataFrame, stand_times_filtered: pd.DataFrame) -> float:
    return get_average_difference(origin_filtered, stand_times_filtered, "origin_time", "departure_stand_actual_time")


def average_taxi_time(
    mfs: pd.DataFrame, standtimes: pd.DataFrame, runways_filtered: pd.DataFrame, departures: bool = True
) -> float:
    merged_df: pd.DataFrame = runways_filtered.merge(mfs[mfs["isdeparture"] == departures], on="gufi")

    return (
        get_average_difference(merged_df, standtimes, "departure_runway_actual_time", "departure_stand_actual_time")
        if departures
        else get_average_difference(merged_df, standtimes, "arrival_stand_actual_time", "arrival_runway_actual_time")
    )


def average_flight_delay(standtimes: pd.DataFrame) -> float:
    """
    Delta between the true time it took to fly to the airport
    and estimated time was supposed to take for the flight to happen
    """
    return get_average_difference_of_columns(standtimes, "arrival_stand_actual_time", "departure_stand_actual_time")


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(df: pd.DataFrame, now: pd.Timestamp, hours: int) -> pd.DataFrame:
    hours_bound: pd.Timestamp = now - timedelta(hours=hours)
    return df.loc[(df.timestamp > hours_bound) & (df.timestamp <= now)]
