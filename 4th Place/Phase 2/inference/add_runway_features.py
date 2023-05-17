#
# Author: Daniil Filienko (?)
#
# extract potential useful features from configs [latest available runways data]
#

from datetime import datetime
import numpy as np

import pandas as pd

airports = ["KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA"]


def add_runway_features(_df: pd.DataFrame, raw_data: pd.DataFrame, airport: str) -> pd.DataFrame:
    features = pd.DataFrame()
    # Filter airport of interest
    current = raw_data.copy()

    # Add fictitious row in a future timestamp
    enlarge_configs = pd.DataFrame(
        {
            "timestamp": [_df["timestamp"].max()],
            "airport_config": [current.iloc[-1]["departure_runways"]],
            "airport_config2": [current.iloc[-1]["arrival_runways"]],
        }
    )

    current = current._append(enlarge_configs)

    # Aggregate at a 15minute time window
    current = current.groupby("timestamp").airport_config.last().reset_index()
    current = (
        current.set_index("timestamp")
        .airport_config2.resample("15min")
        .airport_config.resample("15min")
        .ffill()
        .reset_index()
    )
    current.columns = ["timestamp", "feat_1_cat_airportconfig", "feat_1_cat_airportconfig"]
    current = current[current["feat_1_cat_airportconfig"].isna() == False]

    # Indicate number of active runways in each direction
    current["feat_1_active_departurerunways"] = current["feat_1_cat_airportconfig"].apply(
        lambda x: len(str(x).split(","))
    )

    # Indicate number of active runways in each direction
    current["feat_1_active_departurerunways"] = current["feat_1_cat_airportconfig"].apply(
        lambda x: len(str(x).split(","))
    )

    print(current.head(5))

    # Rolling variables of active runways for arrival and departures
    for i in [4, 8, 12, 16, 20, 24]:
        current[f"feat_1_active_dep_roll_{i}"] = current[f"feat_1_active_departurerunways"].rolling(i).mean()
        current[f"feat_1_max_directions_roll_{i}"] = current["feat_1_max_directions"].rolling(i).mean()
        current[f"feat_1_unique_directions_roll_{i}"] = current["feat_1_unique_directions"].rolling(i).mean()

    runways = []
    possible_runways = np.unique(raw_data[["departure_runways", "arrival_runways"]].values)

    # making a list of active runways
    for r in possible_runways:
        runway = "".join(filter(lambda x: x.isdigit(), r))
        runways.append(runway)

    # Add binary indicator for each runway indicating if they are active in each direction
    for r in runways:
        current[f"feat_1_active_dep_{r}"] = (
            current["feat_1_cat_airportconfig"].apply(lambda x: r in x.split("_A_")[0]).astype(int)
        )
        current[f"feat_1_active_arrival_{r}"] = (
            current["feat_1_cat_airportconfig"].apply(lambda x: r in x.split("_A_")[1]).astype(int)
        )

        current[f"feat_1_active_dep_{r}R"] = (
            current["feat_1_cat_airportconfig"]
            .apply(lambda x: (r in x.split("_A_")[0]) and (r + "L" not in x.split("_A_")[0]))
            .astype(int)
        )
        current[f"feat_1_active_arrival_{r}R"] = (
            current["feat_1_cat_airportconfig"]
            .apply(lambda x: (r in x.split("_A_")[1]) and (r + "L" not in x.split("_A_")[1]))
            .astype(int)
        )

        current[f"feat_1_active_dep_{r}L"] = (
            current["feat_1_cat_airportconfig"]
            .apply(lambda x: (r in x.split("_A_")[0]) and (r + "R" not in x.split("_A_")[0]))
            .astype(int)
        )
        current[f"feat_1_active_arrival_{r}L"] = (
            current["feat_1_cat_airportconfig"]
            .apply(lambda x: (r in x.split("_A_")[1]) and (r + "R" not in x.split("_A_")[1]))
            .astype(int)
        )

        for i in [4, 8, 12, 16, 20, 24]:
            current[f"feat_1_active_dep_{r}_roll_{i}"] = current[f"feat_1_active_dep_{r}"].rolling(i).mean()
            current[f"feat_1_active_arrival_{r}_roll_{i}"] = current[f"feat_1_active_arrival_{r}"].rolling(i).mean()

            current[f"feat_1_active_dep_{r}R_roll_{i}"] = current[f"feat_1_active_dep_{r}R"].rolling(i).mean()
            current[f"feat_1_active_arrival_{r}R_roll_{i}"] = current[f"feat_1_active_arrival_{r}R"].rolling(i).mean()

            current[f"feat_1_active_dep_{r}L_roll_{i}"] = current[f"feat_1_active_dep_{r}L"].rolling(i).mean()
            current[f"feat_1_active_arrival_{r}L_roll_{i}"] = current[f"feat_1_active_arrival_{r}L"].rolling(i).mean()

        # Add categorical features in 25 lookback period of previous configuration
        for i in range(1, 25):
            current[f"feat_1_cat_previous_config_{i}"] = current["feat_1_cat_airportconfig"].shift(i)

        # Extract features based on number of changes in last periods
        for i in [4, 8, 12, 16, 20, 24]:
            current[f"feat_1_nchanges_last_{i}"] = current[
                [f"feat_1_cat_previous_config_{j}" for j in range(1, 25) if j < i]
            ].nunique(axis=1)

        # Insert airport name
        current.insert(0, "airport", airport)

        # Append configuration features with current airport
        features = pd.concat([features, current])

        # Add features of the current airport as global features to the master table
        for i in [4, 8, 12, 16, 20, 24]:
            _df[f"feat_1_{airport}_dep_roll_{i}"] = _df["timestamp"].map(
                dict(zip(current["timestamp"], current[f"feat_1_active_dep_roll_{i}"]))
            )
            _df[f"feat_1_{airport}_arr_roll_{i}"] = _df["timestamp"].map(
                dict(zip(current["timestamp"], current[f"feat_1_active_arrival_roll_{i}"]))
            )
            _df[f"feat_1_{airport}_nchanges_{i}"] = _df["timestamp"].map(
                dict(zip(current["timestamp"], current[f"feat_1_nchanges_last_{i}"]))
            )

    _df = _df.merge(features, how="left", on=["timestamp"])

    return _df


def add_runway_arrival_features(_df: pd.DataFrame, raw_data: pd.DataFrame, airport: str) -> pd.DataFrame:
    """
    Extracts features based on past confirmed runway arrival events
    :param pd.Dataframe _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    features = pd.DataFrame()
    current = raw_data.copy()

    current["flight_ind1"] = current["gufi"].apply(lambda x: x.split(".")[0])
    current["flight_ind2"] = current["gufi"].apply(lambda x: x.split(".")[1])
    current["indicator"] = 1

    for i in [5, 10, 15, 30, 60, 120, 360]:
        current[f"arrivals_last_{i}min"] = current.rolling(f"{i}min", on="timestamp").indicator.sum()

    current["timestamp"] = current["timestamp"].dt.ceil("15min")
    current = current.groupby("timestamp").agg(
        {
            "indicator": "sum",
            "arrivals_last_5min": "last",
            "arrivals_last_10min": "last",
            "arrivals_last_15min": "last",
            "arrivals_last_30min": "last",
            "arrivals_last_60min": "last",
            "arrivals_last_120min": "last",
            "arrivals_last_360min": "last",
            "flight_ind1": ["last", "nunique"],
            "flight_ind2": ["last", "nunique"],
            "arrival_runway": ["last", "nunique"],
        }
    )
    current.columns = ["feat_2_" + c[0] + "_" + c[1] for c in current.columns]
    current.rename(
        columns={
            "feat_2_flight_ind1_last": "feat_2_cat_flight_ind1_last",
            "feat_2_flight_ind2_last": "feat_2_cat_flight_ind2_last",
            "feat_2_arrival_runway_last": "feat_2_cat_arrival_runway_last",
        },
        inplace=True,
    )
    current.reset_index(inplace=True)
    current["airport"] = airport

    print("GOOD")
    features = pd.concat([features, current])

    _df = _df.merge(features, how="left", on=["timestamp"])

    return _df


def add_runway_departure_features(_df: pd.DataFrame, raw_data: pd.DataFrame, airport: str) -> pd.DataFrame:
    """
    Extracts features based on past confirmed runway departure events
    :param pd.Dataframe _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    features = pd.DataFrame()

    current = raw_data.copy()

    current["flight_ind1"] = current["gufi"].apply(lambda x: x.split(".")[0])
    current["flight_ind2"] = current["gufi"].apply(lambda x: x.split(".")[1])
    current["indicator"] = 1

    for i in [5, 10, 15, 30, 60, 120, 360]:
        current[f"deps_last_{i}min"] = current.rolling(f"{i}min", on="timestamp").indicator.sum()

    current["timestamp"] = current["timestamp"].dt.ceil("15min")
    current = current.groupby("timestamp").agg(
        {
            "indicator": "sum",
            "deps_last_5min": "last",
            "deps_last_10min": "last",
            "deps_last_15min": "last",
            "deps_last_30min": "last",
            "deps_last_60min": "last",
            "deps_last_120min": "last",
            "deps_last_360min": "last",
            "flight_ind1": ["last", "nunique"],
            "flight_ind2": ["last", "nunique"],
            "departure_runway": ["last", "nunique"],
        }
    )

    current.columns = ["feat_3_" + c[0] + "_" + c[1] for c in current.columns]
    current.rename(
        columns={
            "feat_3_flight_ind1_last": "feat_3_cat_flight_ind1_last",
            "feat_3_flight_ind2_last": "feat_3_cat_flight_ind2_last",
            "feat_3_departure_runway_last": "feat_3_cat_dep_runway_last",
        },
        inplace=True,
    )

    current.reset_index(inplace=True)
    current["airport"] = airport

    print("DONE")
    features = pd.concat([features, current])

    _df = _df.merge(features, how="left", on=["timestamp"])

    return _df
