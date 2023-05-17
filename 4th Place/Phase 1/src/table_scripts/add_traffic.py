#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

from .feature_engineering import filter_by_timestamp
import pandas as pd


# calculate various traffic measures for airport
def add_traffic(
    now: pd.Timestamp,
    flights_selected: pd.DataFrame,
    latest_etd: pd.DataFrame,
    data_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    mfs = data_tables["mfs"]
    runways = data_tables["runways"]
    standtimes = data_tables["standtimes"]

    runways_filtered_3hr = filter_by_timestamp(runways, now, 3)

    deps_3hr = count_actual_flights(runways_filtered_3hr, departures=True)
    flights_selected["deps_3hr"] = pd.Series(
        [deps_3hr] * len(flights_selected), index=flights_selected.index
    )

    deps_30hr = count_actual_flights(runways, departures=True)
    flights_selected["deps_30hr"] = pd.Series(
        [deps_30hr] * len(flights_selected), index=flights_selected.index
    )

    arrs_3hr = count_actual_flights(runways_filtered_3hr, departures=False)
    flights_selected["arrs_3hr"] = pd.Series(
        [arrs_3hr] * len(flights_selected), index=flights_selected.index
    )

    arrs_30hr = count_actual_flights(runways, departures=False)
    flights_selected["arrs_30hr"] = pd.Series(
        [arrs_30hr] * len(flights_selected), index=flights_selected.index
    )

    # technically is the # of planes whom have arrived at destination airport gate and also departed their origin
    # airport over 30 hours ago, but who cares, it's an important feature regardless
    deps_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="departures")
    flights_selected["deps_taxiing"] = pd.Series(
        [deps_taxiing] * len(flights_selected), index=flights_selected.index
    )

    arrs_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="arrivals")
    flights_selected["arrs_taxiing"] = pd.Series(
        [arrs_taxiing] * len(flights_selected), index=flights_selected.index
    )

    # apply count of expected departures within various windows
    flights_selected["exp_deps_15min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 15), axis=1
    )

    flights_selected["exp_deps_30min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 30), axis=1
    )

    return flights_selected


def count_actual_flights(runways_filtered, departures: bool) -> int:
    if departures:
        runways_filtered = runways_filtered.loc[
            pd.notna(runways_filtered["departure_runway_actual_time"])
        ]
    else:
        runways_filtered = runways_filtered.loc[
            pd.notna(runways_filtered["arrival_runway_actual_time"])
        ]

    return runways_filtered.shape[0]


def count_planes_taxiing(mfs, runways, standtimes, flights: str) -> int:
    mfs = mfs.loc[mfs["isdeparture"] == (flights == "departures")]

    if flights == "departures":
        taxi = pd.merge(
            mfs, standtimes, on="gufi"
        )  # inner join will only result in flights with departure stand times
        taxi = pd.merge(
            taxi, runways, how="left", on="gufi"
        )  # left join leaves blanks for taxiing flights
        taxi = taxi.loc[
            pd.isna(taxi["departure_runway_actual_time"])
        ]  # select the taxiing flights
    elif flights == "arrivals":
        taxi = runways.loc[
            pd.notna(runways["arrival_runway_actual_time"])
        ]  # arrivals are rows with valid time
        taxi = pd.merge(
            taxi, standtimes, how="left", on="gufi"
        )  # left merge with standtime
        taxi = taxi.loc[
            pd.isna(taxi["arrival_stand_actual_time"])
        ]  # empty standtimes mean still taxiing
    else:
        raise RuntimeError("Invalid argument, must specify departures or arrivals")

    return taxi.shape[0]


def count_expected_departures(gufi: str, etd: pd.DataFrame, window: int) -> int:
    time = etd.loc[etd.index == gufi]["departure_runway_estimated_time"].iloc[0]

    lower_bound = time - pd.Timedelta(minutes=window)
    upper_bound = time + pd.Timedelta(minutes=window)

    etd_window = etd.loc[
        (etd["departure_runway_estimated_time"] >= lower_bound)
        & (etd["departure_runway_estimated_time"] <= upper_bound)
    ]

    return etd_window.shape[0]
