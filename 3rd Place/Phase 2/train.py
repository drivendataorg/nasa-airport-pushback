"""Federated training using Flower."""
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List
import math
import time

import flwr as fl
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import pickle
from config import (
    AIRLINES,
    AIRPORTS,
    AIRPORT_MASK,
    NUM_TRAINING_ROUNDS,
    NO_LOAD_PREMADE_PUBLIC_DATA,
    DOWN_SAMPLING_SIZE,
    public_features_directory,
    private_features_directory,
    ann_model_directory,
)

# Merged both util file as only have federated version, no reference centralized version
from utils import (
    get_airline_labels,
    load_private_airline_airport_features,
    load_public_airport_features,
    initialize_model,
    load_airport_labels,
    private_data_model_specific_preprocessing,
    ProcessedPublicFeatures,
    train_public_model,
    check_airline_saved,
)


def train_client_factory(airline: str):
    # Public Features
    airport_public_data_dict = {}
    for airport in AIRPORTS:
        if AIRPORT_MASK[airport]:
            fp = f"{public_features_directory}/processed_public_features_{airport}.pkl"
            with open(fp, "rb") as f:
                airport_public_data_dict[airport] = pickle.load(f)

    # Private Data
    features = []
    labels = []
    if not check_airline_saved(airline):
        for airport in AIRPORTS:
            if AIRPORT_MASK[airport]:
                airport_labels = load_airport_labels(airport, downsample=0)

                airlines = pd.Series(
                    airport_labels.index.get_level_values("gufi").str[:3],
                    index=airport_labels.index,
                )
                if airline not in airlines.values:
                    logger.warning(f"{airline} not present at {airport}.")
                    continue

                airline_labels = get_airline_labels(airport_labels, airline)

                # Downsampling huge data
                if DOWN_SAMPLING_SIZE:
                    if airline_labels.shape[0] > DOWN_SAMPLING_SIZE:
                        airline_labels = airline_labels.sample(n=DOWN_SAMPLING_SIZE)
                        logger.debug(
                            f"Downsampled to {DOWN_SAMPLING_SIZE} data points for {airline} @ {airport}"
                        )

                private_features = load_private_airline_airport_features(
                    airline_labels.index, airline, airport
                )

                public_features = airport_public_data_dict[airport]["X"]

                private_features = private_features.reset_index()
                public_features = public_features.reset_index()
                airline_labels = airline_labels.reset_index()
                airline_features = pd.merge(
                    private_features, public_features, on=["gufi", "timestamp"]
                )
                airline_labels = airline_features.merge(
                    airline_labels, on=["gufi", "timestamp"]
                )

                features.append(airline_features)
                labels.append(airline_labels)

        features = pd.concat(features, axis=0)
        labels = pd.concat(labels, axis=0)

        features, labels = private_data_model_specific_preprocessing(
            features, labels, np.float32
        )

        features.to_csv(
            f"{private_features_directory}/{airline}_features.csv", index=False
        )
        labels.to_csv(f"{private_features_directory}/{airline}_labels.csv", index=False)

    else:
        features_fp = f"{private_features_directory}/{airline}_features.csv"
        labels_fp = f"{private_features_directory}/{airline}_labels.csv"
        features = pd.read_csv(features_fp)
        labels = pd.read_csv(labels_fp)

    x_train_values, x_test_values, y_train_values, y_test_values = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = initialize_model(n_features=int(features.shape[1]))
    model_num = 0
    if model_num:
        fp = f"{ann_model_directory}/federated_weights_epoch_{model_num}.npz"
        weights = np.load(fp)
        weights = [weights[key] for key in weights]
        model.set_weights(weights)

    @dataclass
    class NNTrainClient(fl.client.NumPyClient):
        """NN flower client."""

        airline: str

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            logger.debug(f"Fitting client {self.airline}")
            """
            Need to experiment with:
                - batch_size
                - steps_per_epoch
            """
            epochs = 1
            batch_size = len(x_train_values)
            hist = model.fit(
                x_train_values, y_train_values, epochs=epochs, batch_size=batch_size
            )
            log_file = "train_log.txt"
            with open(log_file, "a") as file:
                start_time = time.time()
                file.write(f"Training Airline: {airline}, EndTime: {start_time}")
                file.write(
                    f"Loss(MSE): {hist.history['loss']}, MAE: {hist.history['mae']}\n"
                )

            # This might be extremely noisy in logs trying to log all weights and biases, leaving for now
            model_parameters = model.get_weights()

            return model_parameters, len(x_train_values), {}

        def evaluate(self, parameters, config):
            # logger.debug(f"Type: {type(parameters)}")
            # logger.debug(f"Val: \n{parameters}")
            logger.debug(f"Evaluating client {self.airline}")
            model.set_weights(parameters)
            mse, mae = model.evaluate(x_test_values, y_test_values)
            logger.debug(f"{self.airline} MSE = {mse:4.4f}")
            logger.debug(f"{self.airline} MAE = {mae:4.4f}")
            log_file = "train_log.txt"
            with open(log_file, "a") as file:
                start_time = time.time()
                file.write(f"Eval Airline: {airline}, EndTime: {start_time}, ")
                file.write(f"Loss(MSE): {mse}, Approx MAE: {mae}\n")
            return mse, len(x_test_values), {"mae": float(mae)}

    return NNTrainClient(airline)


def fit_config(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        min_available_clients,
        on_fit_config_fn,
        min_fit_clients,
        fraction_fit,
        fraction_evaluate,
    ):
        self.training_rounds = 0
        super(SaveModelStrategy, self).__init__(
            min_available_clients=min_available_clients,
            on_evaluate_config_fn=on_fit_config_fn,
            min_fit_clients=min_fit_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
        )

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            logger.info(f"Saving round {server_round} aggregated_ndarrays")
            fp = f"{ann_model_directory}/federated_weights_epoch_{self.training_rounds}.npz"
            self.training_rounds += 1
            np.savez(fp, *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":
    if NO_LOAD_PREMADE_PUBLIC_DATA:
        airport_public_data_dict = {}
        for airport in AIRPORTS:
            if AIRPORT_MASK[airport]:
                airport_public_data_dict[airport] = {}
                airport_labels = load_airport_labels(airport, downsample=False)
                public_features = load_public_airport_features(
                    airport_labels.index, airport
                )
                airport_public_data_dict[airport]["X"] = public_features
                airport_labels = public_features.merge(
                    airport_labels, left_index=True, right_index=True
                )["minutes_until_pushback"].to_frame()
                airport_public_data_dict[airport]["y"] = airport_labels
                with open(
                    f"{public_features_directory}/processed_public_features_{airport}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(airport_public_data_dict[airport], f)

    train_public_model()

    client_resources = {
        "num_cpus": int(os.cpu_count() / 2) + 1,
    }

    strategy = SaveModelStrategy(
        min_available_clients=len(AIRLINES),
        on_fit_config_fn=fit_config,
        min_fit_clients=len(AIRLINES),
        fraction_fit=1,
        fraction_evaluate=1,
    )

    fl.simulation.start_simulation(
        client_fn=train_client_factory,
        client_resources=client_resources,
        clients_ids=AIRLINES,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_TRAINING_ROUNDS),
        ray_init_args={"include_dashboard": False},
    )
