# Import all necesssary modules

import os
import sys
import timeit
import pandas as pd
import numpy as np


from config import *
from logging import DEBUG, INFO
from datetime import datetime, timedelta
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import torch, torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanAbsoluteError
from tqdm import trange, tqdm
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split

import flwr as fl
from flwr.common.typing import Parameters
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from flwr.common import NDArray, NDArrays

import functools
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig

from flwr.common import DisconnectRes, Parameters, ReconnectIns, Scalar
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.server.server import (  # server.py
    reconnect_clients,
    reconnect_client,
    fit_clients,
    fit_client,
    _handle_finished_future_after_fit,
    evaluate_clients,
    evaluate_client,
    _handle_finished_future_after_evaluate,
)
from flwr.common import (  # client.py
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample


# Global constants
client_tree_num = 50
device = "cpu"
