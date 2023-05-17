#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#
import pandas as pd

from .feature_engineering import average_stand_time, average_taxi_time


# calculate various traffic measures for airport
def add_averages(
    flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs: pd.DataFrame = data_tables["mfs"]

    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    origin: pd.DataFrame = data_tables["first_position"].rename(
        columns={"timestamp": "origin_time"}
    )

    standtime_30hr = average_stand_time(origin, standtimes)
    flights_selected["standtime_30hr"] = pd.Series(
        [standtime_30hr] * len(flights_selected), index=flights_selected.index
    )

    dep_taxi_30hr = average_taxi_time(mfs, standtimes, runways)
    flights_selected["dep_taxi_30hr"] = pd.Series(
        [dep_taxi_30hr] * len(flights_selected), index=flights_selected.index
    )

    arr_taxi_30hr = average_taxi_time(mfs, standtimes, runways, departures=False)
    flights_selected["arr_taxi_30hr"] = pd.Series(
        [arr_taxi_30hr] * len(flights_selected), index=flights_selected.index
    )

    return flights_selected
