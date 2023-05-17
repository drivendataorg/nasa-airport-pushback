#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Trevor Tomlin
#
# This script builds a table of training data for a single airport that is hard coded.
#
# It can easily be changed.
#
import gc
import os
import shutil

import pandas as pd
from table_dtype import TableDtype
from table_generation import generate_table
from utils import train_test_split


# _airports - a list of airports which you want to process. EX: ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
# _ROOT - the root
# outputCsvToDir - the path to the directory where you want your csv files to be saved to, using _ROOT by default
def generate(
    _airports: list[str],
    _ROOT: str,
    DATA_DIR: str | None = None,
    outputCsvToDir: str | None = None,
    submission_format: pd.DataFrame | None = None,
    max_rows: int = -1,
) -> None:
    # the path to the directory where data files are stored
    if DATA_DIR is None:
        DATA_DIR: str = os.path.join(_ROOT, "_data")

    # if outputCsvToDir is not given, then use ROOT
    if outputCsvToDir is None:
        outputCsvToDir = DATA_DIR

    for airport in _airports:
        print("Processing", airport)

        # extract features for give airport
        table: dict[str, pd.DataFrame] = generate_table(airport, DATA_DIR, submission_format, max_rows)

        # remove old csv
        out_dirs: dict[str, str] = {
            "train_tables": os.path.join(outputCsvToDir, "train_tables", airport),
            "validation_tables": os.path.join(outputCsvToDir, "validation_tables", airport),
            "full_tables": os.path.join(outputCsvToDir, "full_tables", airport),
        }
        for _out_path in out_dirs.values():
            # remove old csv
            if os.path.exists(_out_path):
                shutil.rmtree(_out_path)
            # and create new folder
            os.mkdir(_out_path)

        for k in table:
            # some int features may be missing due to a lack of information
            table[k] = TableDtype.fix_potential_missing_int_features(table[k])

            # fix index issue
            table[k].reset_index(drop=True, inplace=True)

            # fill the result missing spot with UNK
            for _col in table[k].select_dtypes(include=["category"]).columns:
                table[k][_col] = table[k][_col].cat.add_categories("UNK")
                table[k][_col] = table[k][_col].fillna("UNK").astype(str)

            # fill null
            table[k].fillna("UNK", inplace=True)

            # -- save data ---
            # full
            table[k].sort_values(["gufi", "timestamp"]).to_csv(
                os.path.join(out_dirs["full_tables"], f"{k}_full.csv"), index=False
            )
            # split
            train_test_split(table[k], DATA_DIR, out_dirs, airport, k)

        print("Finished processing", airport)
        print("------------------------------")

        gc.collect()

    print("Done")


if __name__ == "__main__":
    import argparse

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    args: argparse.Namespace = parser.parse_args()

    # airports need to process
    allAirports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in allAirports:
            allAirports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airport_selected}!")

    _ROOT = os.path.join(os.path.dirname(__file__), "..")

    generate(allAirports, _ROOT, max_rows=-1 if args.m is None else int(args.m))
