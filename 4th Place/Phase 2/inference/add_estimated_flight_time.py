#
# Author: Daniil Filienko
#
# calculate and add flight duration information to the general data frame
#

import pandas as pd


def add_estimated_flight_time(
    flights_selected: pd.DataFrame, latest_etd: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    The estimated time it takes for the flight to happen
    Available for the predicted flight
    """
    latest_tfm: pd.DataFrame = data_tables["tfm"].groupby("gufi").last()

    merged_df = pd.merge(flights_selected, latest_tfm, on="gufi")

    merged_df = pd.merge(merged_df, latest_etd, on="gufi")

    # Estimated flight duration for all of the flights within last 30 hours
    merged_df["flight_time"] = (
        merged_df["arrival_runway_estimated_time"] - merged_df["departure_runway_estimated_time"]
    ).dt.total_seconds() / 60

    merged_df["flight_time"] = merged_df["flight_time"].fillna(0)
    merged_df["flight_time"] = merged_df["flight_time"].round()

    flights_selected = pd.merge(flights_selected, merged_df[["flight_time"]], on="gufi", how="left")

    return flights_selected
