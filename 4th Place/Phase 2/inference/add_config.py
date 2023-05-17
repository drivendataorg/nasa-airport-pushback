#
# Author: Jeff Maloney, Kyler Robison, Daniil Filienko
#
# calculate and add airport runway configurations to the data frame
#

import pandas as pd


# find and add current runway configuration
def add_config(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
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

    # flights_selected = add_ratios(flights_selected, "departure_runways")
    # flights_selected = add_ratios(flights_selected, "arrival_runways")

    return flights_selected


def add_ratios(flights_selected: pd.DataFrame, column_name: str):
    new_column_name = column_name + "_ratio"

    flights_selected[column_name] = flights_selected[column_name].astype(str)

    # Compute the mean number of commas in the given column
    mean_num_commas = flights_selected[column_name].str.count(",").mean() + 1

    # Compute the max number of commas in the given column
    max_num_commas = flights_selected[column_name].str.count(",").max() + 1

    # Loop through each row in the DataFrame
    for index, row in flights_selected.iterrows():
        num_commas = row[column_name].count(",")
        if num_commas == 0:
            # If row doesn't contain any commas, fill with mean number of commas divided by maximum number of columns
            flights_selected.at[index, new_column_name] = mean_num_commas / max_num_commas
        else:
            flights_selected.at[index, new_column_name] = num_commas / max_num_commas

    flights_selected[new_column_name] = flights_selected[new_column_name].fillna(0).astype(float)
    flights_selected[new_column_name] = flights_selected[new_column_name].round(1)

    return flights_selected
