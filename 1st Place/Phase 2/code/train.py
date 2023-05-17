import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error

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
END_TIME = "2022-12-02"
END_TRAIN = "2022-09-01"

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


def extract_public_features(airport: str, train=True):
    """
    Function to extract public features for a given airport
    If the features have been already computed once, they are loaded,
    otherwise they are computed across all publicly available data
    sources
    """
    feature_file = os.path.join(
        BASE_PATH, "data", "interim", f"public_{airport}.pickle"
    )

    if os.path.exists(feature_file):
        with open(feature_file, "rb") as f:
            public_features = pickle.load(f)

    else:
        # Read raw public data
        config = pd.read_csv(
            os.path.join(
                BASE_PATH,
                "data",
                "raw",
                "public",
                airport,
                f"{airport}_config.csv.bz2",
            ),
            parse_dates=["timestamp"],
        )
        etd = pd.read_csv(
            os.path.join(
                BASE_PATH, "data", "raw", "public", airport, f"{airport}_etd.csv.bz2",
            ),
            parse_dates=["timestamp", "departure_runway_estimated_time"],
        )
        lamp = pd.read_csv(
            os.path.join(
                BASE_PATH, "data", "raw", "public", airport, f"{airport}_lamp.csv.bz2",
            ),
            parse_dates=["timestamp", "forecast_timestamp"],
        )
        runways = pd.read_csv(
            os.path.join(
                BASE_PATH,
                "data",
                "raw",
                "public",
                airport,
                f"{airport}_runways.csv.bz2",
            ),
            parse_dates=[
                "timestamp",
                "departure_runway_actual_time",
                "arrival_runway_actual_time",
            ],
        )
        if train:
            perimeter = pd.read_csv(
                os.path.join(
                    BASE_PATH, "data", "raw", f"phase2_train_labels_{airport}.csv.bz2",
                ),
                parse_dates=["timestamp"],
            )
        else:
            perimeter = "READ PARTIAL SUBMISSION FORMAT HERE!"

        # Extract features for each data block
        public_features = {}
        public_features["perimeter"] = perimeter
        public_features["etd"] = extract_etd_features(public_features["perimeter"], etd)
        public_features["mom"] = extract_moment_features(public_features["perimeter"])
        public_features["config"] = extract_config_features(
            config, START_TIME, END_TIME
        )
        public_features["lamp"] = extract_weather_features(lamp, START_TIME, END_TIME)
        public_features["runways"] = extract_runway_features(
            runways, START_TIME, END_TIME
        )

        os.makedirs(os.path.expanduser(os.path.dirname(feature_file)), exist_ok=True)
        # Save the resulting public features
        with open(feature_file, "wb") as f:
            pickle.dump(public_features, f)

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
            "raw",
            "private",
            airport,
            f"{airport}_{airline}_mfs.csv.bz2",
        )
    )
    standtimes = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "data",
            "raw",
            "private",
            airport,
            f"{airport}_{airline}_standtimes.csv.bz2",
        ),
        parse_dates=["timestamp", "departure_stand_actual_time"],
    )

    # Load etd data (necessary to extract standtime features)
    etd = pd.read_csv(
        os.path.join(
            BASE_PATH, "data", "raw", "public", airport, f"{airport}_etd.csv.bz2",
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


def initialize_model(airport: str):
    """
    Loads and returns saved model for a given airport if the path exists
    and otherwise initiates it as an empty CatBoostRegressor
    """
    model_path = os.path.join(BASE_PATH, "models", f"{airport}_model")

    model = CatBoostRegressor(
        eta=0.01,
        depth=7,
        rsm=1,
        subsample=0.8,
        max_leaves=21,
        l2_leaf_reg=3,
        min_data_in_leaf=50,
        n_estimators=5000,
        task_type="CPU",
        thread_count=-1,
        grow_policy="Lossguide",
        has_time=True,
        random_seed=4,
        loss_function="MAE",
        boosting_type="Plain",
        max_ctr_complexity=12,
        bootstrap_type="Bernoulli",
    )
    print("Best iteration", model.best_iteration_)

    if os.path.exists(model_path):
        model.load_model(model_path)
        print("Best iteration after loading", model.best_iteration_)

    return model


def save_weights(model, airport):
    """
    Function that saves the model artifacts for a given model and airport specification
    """
    model_path = os.path.join(BASE_PATH, "models", f"{airport}_model")
    model.save_model(model_path)


def train_client_factory(airport: str, airline: str):
    """
    Main code for the flower framework to trigger the master creation and
    model training for each client instance - being at the airline level
    """
    try:
        # Load public features for the specified airport and all airlines
        public_features = extract_public_features(airport, train=True)

        # Load private features for the specified airline and airport pair
        private_features = extract_private_features(public_features, airport, airline)

        # Filter perimeter to stick only to current airline
        perimeter = public_features["perimeter"][
            public_features["perimeter"]["gufi"].str[:3] == airline
        ]

        # Combine all features into a master table
        master = build_master(
            base=perimeter,
            mfs=private_features["mfs"],
            config=public_features["config"],
            etd=public_features["etd"],
            mom=public_features["mom"],
            weather=public_features["lamp"],
            runways=public_features["runways"],
            standtimes=private_features["standtimes"],
        )

        # Define the train validation split for the current airline slice
        feat_names = [
            c
            for c in master.columns
            if c not in ["timestamp", "gufi", "minutes_until_pushback"]
        ]
        cat_features = [i for i in range(len(feat_names)) if "_cat_" in feat_names[i]]
        x_train = master[master["timestamp"] < END_TRAIN][feat_names]
        x_val = master[master["timestamp"] >= END_TRAIN][feat_names]
        y_train = master[master["timestamp"] < END_TRAIN]["minutes_until_pushback"]
        y_val = master[master["timestamp"] >= END_TRAIN]["minutes_until_pushback"]

        @dataclass
        class CatBoostRegressionTrainClient(fl.client.NumPyClient):
            """CatBoost regression flower client"""

            airline: str

            def fit(self, parameters=None, config=None):
                model_path = os.path.join(BASE_PATH, "models", f"{airport}_model")
                previous_model = initialize_model(airport)

                logger.info(f"Fitting airport {airport} with {airline} data")
                logger.info(
                    f"Best iteration of previous model {previous_model.best_iteration_}"
                )
                logger.info(
                    f"Training with {x_train.shape} and validating on {x_val.shape}"
                )

                model = CatBoostRegressor(
                    eta=0.01,
                    depth=7,
                    rsm=1,
                    subsample=0.8,
                    max_leaves=21,
                    l2_leaf_reg=3,
                    min_data_in_leaf=50,
                    n_estimators=5000,
                    task_type="CPU",
                    thread_count=-1,
                    grow_policy="Lossguide",
                    has_time=True,
                    random_seed=4,
                    loss_function="MAE",
                    boosting_type="Plain",
                    max_ctr_complexity=12,
                    bootstrap_type="Bernoulli",
                )

                model.fit(
                    x_train,
                    y_train,
                    eval_set=(x_val, y_val),
                    use_best_model=True,
                    verbose=200,
                    cat_features=cat_features,
                    early_stopping_rounds=20,
                    init_model=previous_model if os.path.exists(model_path) else None,
                )

                save_weights(model, airport)

                return np.array([[1, 2, 3], [4, 5, 6]], np.int32), len(y_train), {}

            def evaluate(self, parameters, config):
                logger.info(f"Evaluating client {self.airline}")
                model = initialize_model(airport)
                predictions = model.predict(x_val)
                loss = mean_absolute_error(y_val, predictions)
                logger.info(f"{self.airline} loss = {loss:4.4f}")
                return loss, len(y_val), {}

        return CatBoostRegressionTrainClient(airline)

    except:

        @dataclass
        class DummyTrainClient(fl.client.NumPyClient):
            """Dummy flower client"""

            airline: str

            def fit(self, parameters=None, config=None):

                return np.array([[1, 2, 3], [4, 5, 6]], np.int32), 100, {}

            def evaluate(self, parameters, config):

                return 100, 100, {}

        return DummyTrainClient(airline)


def fit_config(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


if __name__ == "__main__":
    """
    Main entrypoint to train and save the model artifact for a given airport
    Example execution: python3.10 train.py "airport_name"
    """

    airport = sys.argv[1]

    client_resources = {"num_cpus": 1}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=len(AIRLINES),
        min_evaluate_clients=len(AIRLINES),
        min_available_clients=len(AIRLINES),
    )

    fl.simulation.start_simulation(
        client_fn=lambda x: train_client_factory(airport, x),
        client_resources=client_resources,
        clients_ids=AIRLINES,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=1),
        ray_init_args={"include_dashboard": False, "num_cpus": 1},
    )
