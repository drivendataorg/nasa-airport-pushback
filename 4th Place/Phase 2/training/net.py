import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(features), 100)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(len(features), 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
