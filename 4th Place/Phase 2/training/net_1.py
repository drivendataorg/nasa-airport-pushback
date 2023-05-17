import torch
import torch.nn as nn
from config import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.normalization = nn.LayerNorm(features_len)  # Corresponds to tf.keras.layers.Normalization(axis=-1)

        self.fc1 = nn.Linear(
            len(features), 32
        )  # Corresponds to tf.keras.layers.Dense(32, activation="relu")
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(
            32, 64
        )  # Corresponds to tf.keras.layers.Dense(64, activation="relu")
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(
            64, 64
        )  # Corresponds to tf.keras.layers.Dense(64, activation="relu")
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)  # Corresponds to tf.keras.layers.Dense(1)

    def forward(self, x):
        # x = self.normalization(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
