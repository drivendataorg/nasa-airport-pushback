#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Daniil Filienko
# A set of helper functions
#

import os

import pandas as pd


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


# specify an airport to split for only one, otherwise a split for all airports will be executed
def train_test_split(table: pd.DataFrame, DATA_DIR: str, dirs: dict[str, str], airport: str, _k: str) -> None:
    val_data: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "submission_format.csv"))

    # If there is a specific airport then we are only interested in those rows
    if airport != "ALL":
        val_data = val_data[val_data.airport == airport]

    _gufi = val_data.gufi.unique()
    test_data: pd.DataFrame = table[table.gufi.isin(_gufi)]
    train_data: pd.DataFrame = table[~table.gufi.isin(_gufi)]

    # replace these paths with any desired ones if necessary
    train_data.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(dirs["train_tables"], f"{_k}_train.csv"), index=False
    )
    test_data.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(dirs["validation_tables"], f"{_k}_validation.csv"), index=False
    )


# read the data from all airports and return a unified total table
def get_inference_data(DATA_DIR: str, airlines: list[str], airports: list[str]) -> pd.DataFrame:
    all_airlines_val: list[pd.DataFrame] = []
    # Iterate over each possible airports per airline
    for airline in airlines:
        airline_val_dfs: list[pd.DataFrame] = []
        for airport in airports:
            _csv_path: str = os.path.join(DATA_DIR, "validation_tables", airport, f"{airline}_validation.csv")
            if os.path.exists(_csv_path):
                airline_val_dfs.append(
                    pd.read_csv(
                        f"{DATA_DIR}/validation_tables/{airport}/{airline}_validation.csv",
                        parse_dates=["timestamp"],
                        dtype={"precip": str},
                    )
                )
            else:
                print(f"Skip validation {airport}-{airline} because csv file does not exist")
        if len(airline_val_dfs) == 0:
            # if no data for an airine present, skip
            print(f"Warning: validation data for {airline} does not exists in any validation data!")
            print("This could be intentional. If not, please double check!")
            continue

        # Create an airline level df
        airline_df: pd.DataFrame = pd.concat(airline_val_dfs)

        if airline_df.shape[0] == 0:
            print("************************************************************")
            print(f"Warning: validation data frame for {airline} is empty!")
            print("This may not be intentional. Please double check!")
            print("************************************************************")
            continue
        # Add all airlines into one total list
        all_airlines_val.append(airline_df)

    # Concat all airlines into one dataframe
    return pd.concat(all_airlines_val)


airlines: list[str] = [
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
]

features = [
    "airline",
    "airport",
    "deps_3hr",
    "deps_30hr",
    "deps_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    "1h_ETDP",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "weekday",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "cloud",
    "lightning_prob",
    "gufi_timestamp_until_etd",
]

encoded_columns = [
    "cloud",
    "lightning_prob",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "airport",
    "airline",
]

int_columns = [
    "deps_3hr",
    "deps_30hr",
    "deps_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "minute",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "minutes_until_etd",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "gufi_timestamp_until_etd",
]
