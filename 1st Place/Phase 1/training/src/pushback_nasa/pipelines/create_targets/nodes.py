"""
This is a boilerplate pipeline 'create_targets'
generated using Kedro 0.18.4
"""

import pandas as pd
from tqdm import tqdm


def create_targets(
    standtimes: pd.DataFrame, mfs: pd.DataFrame, start_time: str, end_time: str
) -> pd.DataFrame:
    # Define start and end times
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # Filter standtimes for the date range selected
    standtimes = standtimes[
        (standtimes["timestamp"] >= start_time) & (standtimes["timestamp"] <= end_time)
    ]

    # Filter standtimes for the departures only
    gufi_departures = mfs[mfs["isdeparture"] == True]["gufi"].unique()
    standtimes = standtimes[standtimes["gufi"].isin(gufi_departures)]

    # Map airplanes to actual stand times
    dep_mapping = dict(
        zip(standtimes["gufi"], standtimes["departure_stand_actual_time"])
    )

    # Iterate through departure mapping
    targets = []
    for k in tqdm(dep_mapping.keys()):
        if not pd.isnull(dep_mapping[k]):
            current = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        (dep_mapping[k] - pd.Timedelta("300min")).floor("15min"),
                        dep_mapping[k].floor("15min"),
                        freq="15min",
                    )
                }
            )
            current["gufi"] = k
            current["minutes_until_pushback"] = (
                dep_mapping[k] - current["timestamp"]
            ).dt.seconds / 60
            targets.append(current)

    targets = pd.concat(targets, ignore_index=True)

    return targets
