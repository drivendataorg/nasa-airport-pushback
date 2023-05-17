import pandas as pd
from .mytools import *


def load_all_airports() -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    train_datasets: list[pd.DataFrame] = []
    test_datasets: list[pd.DataFrame] = []

    for airline in AIRLINES:
        if any_ds_exists(airline):
            train_ds, test_ds = get_train_and_test_ds("ALL", airline)
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)

    return train_datasets, test_datasets
