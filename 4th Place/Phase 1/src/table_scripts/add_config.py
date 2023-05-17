#
# Author: Jeff Maloney, Kyler Robison, Daniil Filienko
#
# calculate and add airport runway configurations to the data frame
#

import pandas as pd


# find and add current runway configuration
def add_config(
    flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    # filter features to 30 hours before prediction time to prediction time and save as a copy
    config: pd.DataFrame = data_tables["config"]

    # most recent row of airport configuration
    if config.shape[0] > 0:
        departure_runways = config["departure_runways"].iloc[0]
        arrival_runways = config["arrival_runways"].iloc[0]
    else:
        departure_runways = "UNK"
        arrival_runways = "UNK"

    # add new column for which departure runways are in use at this timestamp
    flights_selected["departure_runways"] = pd.Series(
        [departure_runways] * len(flights_selected), index=flights_selected.index
    )
    flights_selected["arrival_runways"] = pd.Series(
        [arrival_runways] * len(flights_selected), index=flights_selected.index
    )

    return flights_selected
