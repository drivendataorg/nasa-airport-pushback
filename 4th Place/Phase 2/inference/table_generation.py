#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Daniil Filienko
# generate the full table for a specific airport
#

import feature_engineering
import pandas as pd
from add_averages import add_averages, add_averages_private
from add_config import add_config
from add_date import add_date_features
from add_etd import add_etd
from add_etd_features import add_etd_features
from add_lamp import add_lamp
from add_traffic import add_traffic, add_traffic_private
from extract_gufi_features import extract_and_add_gufi_features
from feature_tables import get_public_feature_tables, get_private_feature_tables, PrivateAirlineFeature
from utils import get_csv_path


def _process_public_features(filtered_table: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table = filtered_table.reset_index(drop=True)

    # get current time
    now: pd.Timestamp = filtered_table.timestamp.iloc[0]

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    data_tables = filter_tables(now, data_tables)

    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    # add features
    filtered_table = add_etd(filtered_table, latest_etd)
    filtered_table = add_averages(now, filtered_table, latest_etd, data_tables)
    filtered_table = add_traffic(now, filtered_table, latest_etd, data_tables)
    filtered_table = add_config(filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def _process_airline_private(
    filtered_table: pd.DataFrame,
    airline_private_data: PrivateAirlineFeature,
    public_feature_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table = filtered_table.reset_index(drop=True)

    # get current time
    now: pd.Timestamp = filtered_table.timestamp.iloc[0]

    # get the latest ETD for each flight
    runways: pd.DataFrame = feature_engineering.filter_by_timestamp(public_feature_tables["runways"], now, 30)

    # add features
    filtered_table = add_averages_private(
        filtered_table, airline_private_data.mfs, runways, airline_private_data.standtimes
    )
    filtered_table = add_traffic_private(
        filtered_table, airline_private_data.mfs, runways, airline_private_data.standtimes
    )

    return filtered_table


def filter_tables(now: pd.Timestamp, data_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    new_dict: dict[str, pd.DataFrame] = {}

    for key in data_tables:
        if not key.endswith("mfs"):
            new_dict[key] = feature_engineering.filter_by_timestamp(data_tables[key], now, 30)

    new_dict["mfs"] = filter_mfs(data_tables["mfs"], new_dict["standtimes"])

    return new_dict


def filter_mfs(mfs: pd.DataFrame, standtimes: pd.DataFrame) -> pd.DataFrame:
    return mfs.loc[mfs["gufi"].isin(standtimes["gufi"])]


def generate_table(
    _airport: str, data_dir: str, submission_format: pd.DataFrame | None = None, max_rows: int = -1
) -> dict[str, pd.DataFrame]:
    from pandarallel import pandarallel  # type: ignore

    pandarallel.initialize(verbose=1, progress_bar=True)

    # read train labels for given airport
    _df: pd.DataFrame = (
        submission_format.loc[submission_format.airport == _airport]
        if submission_format is not None
        else pd.read_csv(
            get_csv_path(data_dir, "train_labels_phase2", f"phase2_train_labels_{_airport}.csv"),
            parse_dates=["timestamp"],
        )
    )

    # if you want to select only a certain amount of row
    if max_rows > 0:
        _df = _df[:max_rows]

    # define list of data tables to load and use for each airport
    public_feature_tables: dict[str, pd.DataFrame] = get_public_feature_tables(data_dir, _airport)

    # process all public prediction times in parallel
    _df = _df.groupby("timestamp").parallel_apply(lambda x: _process_public_features(x, public_feature_tables))

    # extract and add mfs information
    _df = extract_and_add_gufi_features(_df)

    # Add runway information
    # _df = _df.merge(public_feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # extract holiday features
    _df = add_date_features(_df)

    # Add additional etd features
    _df = add_etd_features(_df, public_feature_tables["etd"])

    # Add mfs information
    # _df = _df.merge(public_feature_tables["mfs"], how="left", on="gufi")

    # create a dictionary that store all airport data frames
    all_df: dict[str, pd.DataFrame] = {}

    # load private data
    private_feature_tables: dict[str, PrivateAirlineFeature] = get_private_feature_tables(data_dir, _airport)

    # process all private prediction times in parallel
    process_index: int = 1
    for key in private_feature_tables:
        airline_df = _df[_df.airline == key]
        if len(airline_df) > 0:
            print(f"\nProcessing private data for ({process_index}/25):", key)

            if len(private_feature_tables[key].mfs) > 0:
                all_df[key] = airline_df.merge(private_feature_tables[key].mfs, how="left", on="gufi")
        else:
            print(f"\nNo private data for ({process_index}/25):", key)
        process_index += 1

    # set _df as airport global public
    all_df["PUBLIC"] = _df

    return all_df
