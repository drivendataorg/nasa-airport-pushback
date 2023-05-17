import argparse
import copy
import os
import pickle
import sys
import time

import flwr as fl
import matplotlib.pyplot as plt
from network import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import *

data_dict = extract_all_airlines_data()
available_airlines = [val for val in data_dict.keys() if val != "all_features"]
train_test_data_dict = {}
SAMPLE_FRAC = 0.01  # we downsampling to 2%

for airline in available_airlines:
    pd00 = data_dict[airline].sample(frac=SAMPLE_FRAC)
    features = data_dict["all_features"]
    x_train, x_test, y_train, y_test = train_test_split(
        pd00[features].values, pd00["minutes_until_pushback"].values, test_size=0.5
    )
    print(airline, x_train.shape)
    train_tensor = TensorDataset(
        torch.Tensor(x_train), torch.Tensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(dataset=train_tensor, batch_size=32, shuffle=True)
    num_samples = len(x_train)
    train_test_data_dict[airline] = train_loader, x_test, y_test, num_samples, features


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, airline, net, trainloader, x_test, y_test, num_samples):
        self.airline = airline
        self.net = net
        self.trainloader = trainloader
        self.x_test = torch.Tensor(x_test)
        self.y_test = torch.Tensor(y_test).reshape(-1, 1)
        self.num_samples = num_samples

    def get_parameters(self, config):
        print(f"[Client {self.airline}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.airline}] fit, config: {config}")
        set_parameters(self.net, parameters)
        self.net, _ = train_eval(
            self.net,
            self.trainloader,
            self.x_test,
            self.y_test,
            options={"epochs": 40, "lr": 5 * 1e-4, "eval": False},
        )
        return get_parameters(self.net), self.num_samples, {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.airline}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        y_pred = self.net(self.x_test)
        criterion = nn.L1Loss()
        mae = criterion(y_pred, self.y_test).item()
        return mae, len(self.x_test), {"mae": mae}


def numpyclient_fn(airline):
    train_loader, x_test, y_test, num_samples, features = train_test_data_dict[airline]
    net = MLP(len(features), 4)

    return FlowerNumPyClient(airline, net, train_loader, x_test, y_test, num_samples)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            np.savez("./models/federated_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def fit_config(server_round: int):
    """Send round number to client."""
    return {"server_round": server_round}


if __name__ == "__main__":
    client_resources = {
        "num_cpus": int(os.cpu_count() / 2) + 1,
    }

    strategy = SaveModelStrategy(
        min_available_clients=len(available_airlines), on_fit_config_fn=fit_config,
    )

    fl.simulation.start_simulation(
        client_fn=numpyclient_fn,
        client_resources=client_resources,
        clients_ids=available_airlines,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
        ray_init_args={"include_dashboard": False},
    )
