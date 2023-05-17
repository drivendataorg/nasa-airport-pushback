#
# Author: Kyler Robison
#
# calculate and add etd information to the data frame
#

import pandas as pd


# calculate etd
def add_etd(flights_selected: pd.DataFrame, latest_etd: pd.DataFrame) -> pd.DataFrame:
    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = flights_selected.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to flights_selected that represents minutes until pushback
    flights_selected["minutes_until_etd"] = (
        (
            departure_runway_estimated_time - flights_selected.timestamp
        ).dt.total_seconds()
        / 60
    ).astype(int)

    return flights_selected
