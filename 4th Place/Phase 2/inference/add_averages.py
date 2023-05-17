#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import pandas as pd
from datetime import timedelta


# calculate various traffic measures for airport
def add_averages(
    now: pd.Timestamp, flights_selected: pd.DataFrame, latest_etd: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs: pd.DataFrame = data_tables["mfs"]

    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    origin: pd.DataFrame = data_tables["first_position"].rename(columns={"timestamp": "origin_time"})

    # 30 hour features, no need to filter
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways)
    flights_selected["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)

    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes)
    flights_selected["standtime_30hr"] = pd.Series(
        [standtime_30hr] * len(flights_selected), index=flights_selected.index
    )

    dep_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)
    flights_selected["dep_taxi_30hr"] = pd.Series([dep_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    # 3 hour features
    latest_etd = feature_engineering.filter_by_timestamp(latest_etd, now, 3)
    runways = feature_engineering.filter_by_timestamp(runways, now, 3)
    standtimes = feature_engineering.filter_by_timestamp(standtimes, now, 3)
    origin = origin.loc[(origin.origin_time > now - timedelta(hours=3)) & (origin.origin_time <= now)]

    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways)
    flights_selected["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes)
    flights_selected["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    dep_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)
    flights_selected["dep_taxi_3hr"] = pd.Series([dep_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    # 1 hour features
    latest_etd = feature_engineering.filter_by_timestamp(latest_etd, now, 1)
    standtimes = feature_engineering.filter_by_timestamp(standtimes, now, 1)
    PDd_1hr = feature_engineering.average_departure_delay(latest_etd, standtimes, "departure_stand_actual_time")
    flights_selected["1h_ETDP"] = pd.Series([PDd_1hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected


# calculate various traffic measures for airport (with private features)
def add_averages_private(
    flights_selected: pd.DataFrame, private_mfs: pd.DataFrame, runways: pd.DataFrame, private_standtimes: pd.DataFrame
) -> pd.DataFrame:
    arr_taxi_30hr = feature_engineering.average_taxi_time(private_mfs, private_standtimes, runways, departures=False)
    flights_selected["arr_taxi_30hr"] = pd.Series([arr_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    arr_taxi_3hr = feature_engineering.average_taxi_time(private_mfs, private_standtimes, runways, departures=False)
    flights_selected["arr_taxi_3hr"] = pd.Series([arr_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected
