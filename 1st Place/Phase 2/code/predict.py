import os
import time

import pandas as pd
from catboost import CatBoostRegressor
from utilities import (
    adjust_master,
    extract_config_features,
    extract_etd_features,
    extract_mfs_features,
    extract_moment_features,
    extract_runway_features,
    extract_standtime_features,
    extract_weather_features,
)

# Define global variables
BASE_PATH = ".."

START_TIME = "2020-11-01"
END_TIME = "2024-12-02"

AIRLINES = [
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

AIRPORTS = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]


def load_models():

    models = {}
    for airport in AIRPORTS:
        current_model = CatBoostRegressor()
        current_model.load_model(os.path.join(BASE_PATH, "models", f"{airport}_model"))
        models[airport] = current_model

    return models


def extract_public_features(perimeter: pd.DataFrame, airport: str):
    """Function to extract public features for a given airport."""
    config = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "test", "public", airport, f"{airport}_config.csv.bz2"
        ),
        parse_dates=["timestamp"],
    )
    etd = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "test", "public", airport, f"{airport}_etd.csv.bz2"
        ),
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )
    lamp = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "test", "public", airport, f"{airport}_lamp.csv.bz2"
        ),
        parse_dates=["timestamp", "forecast_timestamp"],
    )
    runways = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "test", "public", airport, f"{airport}_runways.csv.bz2"
        ),
        parse_dates=[
            "timestamp",
            "departure_runway_actual_time",
            "arrival_runway_actual_time",
        ],
    )

    # Extract features for each data block
    public_features = {}
    public_features["perimeter"] = perimeter
    public_features["etd"] = extract_etd_features(public_features["perimeter"], etd)
    public_features["mom"] = extract_moment_features(public_features["perimeter"])
    public_features["config"] = extract_config_features(config, START_TIME, END_TIME)
    public_features["lamp"] = extract_weather_features(lamp, START_TIME, END_TIME)
    public_features["runways"] = extract_runway_features(runways, START_TIME, END_TIME)

    return public_features


def extract_private_features(public_features, airport: str, airline: str):
    """
    Function to retrieve private features for a combination of airport and airline
    """

    # Load raw private data
    mfs = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "data",
            "test",
            "private",
            airport,
            f"{airport}_{airline}_mfs.csv.bz2",
        )
    )
    standtimes = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "data",
            "test",
            "private",
            airport,
            f"{airport}_{airline}_standtimes.csv.bz2",
        ),
        parse_dates=["timestamp", "departure_stand_actual_time"],
    )

    # Load etd data (necessary to extract standtime features)
    etd = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "test", "public", airport, f"{airport}_etd.csv.bz2"
        ),
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )

    # Extract features from private data blocks
    private_features = {}
    private_features["mfs"] = extract_mfs_features(public_features["perimeter"], mfs)
    private_features["standtimes"] = extract_standtime_features(
        public_features["perimeter"], standtimes, etd
    )

    return private_features


def build_master(
    base: pd.DataFrame,
    mfs: pd.DataFrame,
    config: pd.DataFrame,
    etd: pd.DataFrame,
    mom: pd.DataFrame,
    weather: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to merge all features into a single master dataframe at a gufi-timestamp level
    """
    # Select columns from base
    base = base[["gufi", "timestamp", "minutes_until_pushback"]]

    # Merge all feature blocks in a single master table
    master = base.merge(mfs, how="left", on=["gufi", "timestamp"])
    master = master.merge(config, how="left", on="timestamp")
    master = master.merge(etd, how="left", on=["gufi", "timestamp"])
    master = master.merge(mom, how="left", on="timestamp")
    master = master.merge(weather, how="left", on="timestamp")
    master = master.merge(runways, how="left", on="timestamp")
    master = master.merge(standtimes, how="left", on="timestamp")
    master = adjust_master(master)

    return master


if __name__ == "__main__":
    """
    Main entrypoint to test and save the predictions
    """

    # Load saved model artifacts
    models = load_models()

    # Load submission format
    sub_format = pd.read_csv(
        os.path.join(BASE_PATH, "data", "test", "submission_format.csv.bz2"),
        parse_dates=["timestamp"],
    )

    # Define placeholder to keep the predictions
    predictions = pd.DataFrame()

    for airport in AIRPORTS:
        print(time.ctime(), f"Generating predictions for {airport}")

        # Define perimeter for the current airport
        perimeter = sub_format[sub_format["airport"] == airport]

        # Load public features for the specified airport and all airlines
        public_features = extract_public_features(perimeter, airport)

        # Define airport master placeholder
        airport_master = pd.DataFrame()

        for airline in AIRLINES:
            print(time.ctime(), f"Generating predictions for airline {airline}")

            try:
                # Load private features for the specified airline and airport pair
                private_features = extract_private_features(
                    public_features, airport, airline
                )

                # Combine all features into a master table
                master = build_master(
                    base=perimeter[perimeter["gufi"].str[:3] == airline],
                    mfs=private_features["mfs"],
                    config=public_features["config"],
                    etd=public_features["etd"],
                    mom=public_features["mom"],
                    weather=public_features["lamp"],
                    runways=public_features["runways"],
                    standtimes=private_features["standtimes"],
                )

                # Concatenate into airport data
                airport_master = pd.concat([airport_master, master])

                print(airport_master.shape)

            except:
                print(time.ctime(), f"Failed for airline {airline}")

        # Retrieve predictions for the given airport
        airport_master["minutes_until_pushback"] = models[airport].predict(
            airport_master[models[airport].feature_names_]
        )

        # Append selected columns into submission file
        predictions = pd.concat(
            [
                predictions,
                airport_master[["gufi", "timestamp", "minutes_until_pushback"]],
            ]
        )

    # Append minutes until pushback to the sub_format
    sub_format.drop(columns="minutes_until_pushback", inplace=True)
    sub_format = sub_format.merge(predictions, how="left", on=["gufi", "timestamp"])

    # Clip values and map to integers
    sub_format["minutes_until_pushback"] = (
        sub_format["minutes_until_pushback"].clip(1, 299).astype(int)
    )

    # Save the submission with all the predictions
    sub_format.to_csv(
        os.path.join(BASE_PATH, "data", "processed", "test_predictions.csv"),
        index=False,
    )
