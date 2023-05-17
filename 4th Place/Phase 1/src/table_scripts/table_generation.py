#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Daniil Filienko
# generate the full table for a specific airport
#
import multiprocessing
import os
from functools import partial

import pandas as pd
from tqdm import tqdm

from .add_averages import add_averages
from .add_config import add_config
from .add_date import add_date_features
from .add_etd import add_etd
from .add_lamp import add_lamp
from .add_traffic import add_traffic
from .extract_gufi_features import extract_and_add_gufi_features
from .feature_engineering import filter_by_timestamp
from .table_dtype import TableDtype


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


def process_timestamp(
    now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(
        drop=True
    )

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    data_tables = filter_tables(now, data_tables)

    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    # add features
    filtered_table = add_etd(filtered_table, latest_etd)
    filtered_table = add_averages(filtered_table, data_tables)
    filtered_table = add_traffic(now, filtered_table, latest_etd, data_tables)
    filtered_table = add_config(filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def filter_tables(
    now: pd.Timestamp, data_tables: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    new_dict = {}

    for key in data_tables:
        if key != "mfs":
            new_dict[key] = filter_by_timestamp(data_tables[key], now, 30)

    new_dict["mfs"] = filter_mfs(data_tables["mfs"], new_dict["standtimes"])

    return new_dict


def filter_mfs(mfs: pd.DataFrame, standtimes) -> pd.DataFrame:
    gufis_wanted = standtimes["gufi"]
    mfs_filtered = mfs.loc[mfs["gufi"].isin(gufis_wanted)]
    return mfs_filtered


def generate_table(_airport: str, data_dir: str, max_rows: int = -1) -> pd.DataFrame:
    # read train labels for given airport
    _df: pd.DataFrame = pd.read_csv(
        get_csv_path(
            data_dir,
            f"train_labels_prescreened",
            f"prescreened_train_labels_{_airport}.csv",
        ),
        parse_dates=["timestamp"],
    )

    # table = table.drop_duplicates(subset=["gufi"])

    # if you want to select only a certain amount of row
    if max_rows > 0:
        _df = _df[:max_rows]

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_etd.csv"),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp"),
        "config": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_config.csv"),
            parse_dates=["timestamp"],
        ).sort_values("timestamp", ascending=False),
        "first_position": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_first_position.csv"),
            parse_dates=["timestamp"],
        ),
        "lamp": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_lamp.csv"),
            parse_dates=["timestamp", "forecast_timestamp"],
        )
        .set_index("timestamp", drop=False)
        .sort_index(),
        "runways": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_runways.csv"),
            parse_dates=[
                "timestamp",
                "departure_runway_actual_time",
                "arrival_runway_actual_time",
            ],
        ),
        "standtimes": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_standtimes.csv"),
            parse_dates=[
                "timestamp",
                "departure_stand_actual_time",
                "arrival_stand_actual_time",
            ],
        ),
        "mfs": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_mfs.csv"),
            dtype={"major_carrier": str},
        ),
    }

    return add_all_features(_df, feature_tables)


def add_all_features(
    _df: pd.DataFrame, feature_tables: dict[str, pd.DataFrame], silent: bool = False
):
    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(process_timestamp, flights=_df, data_tables=feature_tables)
        unique_timestamp = _df.timestamp.unique()
        inputs = zip(pd.to_datetime(unique_timestamp))
        timestamp_tables: list[pd.DataFrame] = executor.starmap(
            fn, tqdm(inputs, total=len(unique_timestamp), disable=silent)
        )

    # concatenate individual prediction times to a single dataframe
    _df = pd.concat(timestamp_tables, ignore_index=True)

    # Add runway information
    # _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # extract and add mfs information
    _df = extract_and_add_gufi_features(_df)

    # extract holiday features
    _df = add_date_features(_df)

    # Add mfs information
    _df = _df.merge(feature_tables["mfs"].fillna("UNK"), how="left", on="gufi")

    # some int features may be missing due to a lack of information
    _df = TableDtype.fix_potential_missing_int_features(_df)

    return _df
