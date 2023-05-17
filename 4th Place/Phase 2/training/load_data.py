import pandas as pd
from config import *
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from airline_dataset import *
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from functools import partial
import pickle


def load_airports(airport: str):
    train_loaders = []
    test_loaders = []

    data = defaultdict()

    encoder = partial(
        OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoders = defaultdict(encoder)

    for airline in AIRLINES:
        try:
            dfs = pd.read_csv(
                f"{ROOT}/full_tables/{airport}/{airline}_full.csv",
                parse_dates=["timestamp"],
            )
        except FileNotFoundError:
            continue

        if len(dfs) == 0:
            continue

        try:
            gufi_groups = dfs["gufi"]

            splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
            train_indices, test_indices = next(splitter.split(dfs, groups=gufi_groups))
            train_dfs = dfs.iloc[train_indices]
            test_dfs = dfs.iloc[test_indices]

        except ValueError:
            continue

        X_train = train_dfs[features].copy()
        y_train = train_dfs["minutes_until_pushback"].copy()
        X_test = test_dfs[features].copy()
        y_test = test_dfs["minutes_until_pushback"].copy()

        data[airline] = defaultdict()
        data[airline]["X_train"] = X_train
        data[airline]["X_test"] = X_test
        data[airline]["y_train"] = y_train
        data[airline]["y_test"] = y_test

        for column in encoded_columns:
            try:
                encoders[column].fit(X_train[[column]])
            except Exception as e:
                print(e)
                print(column)

    for airline in data.keys():
        X_train = data[airline]["X_train"]
        X_test = data[airline]["X_test"]
        y_train = data[airline]["y_train"]
        y_test = data[airline]["y_test"]

        for column in encoded_columns:
            try:
                X_train[column] = encoders[column].transform(X_train[[column]])
                X_test[column] = encoders[column].transform(X_test[[column]])
            except Exception as e:
                print(e)
                print(column)

        for column in X_train.columns:
            if column not in encoded_columns:
                X_train[column] = (X_train[column] - X_train[column].mean()) / X_train[
                    column
                ].std()
                X_test[column] = (X_test[column] - X_test[column].mean()) / X_test[
                    column
                ].std()

        X_train = X_train.to_numpy(dtype=np.float32, copy=True)
        X_test = X_test.to_numpy(dtype=np.float32, copy=True)
        y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        X_train.flags.writeable = True
        y_train.flags.writeable = True
        X_test.flags.writeable = True
        y_test.flags.writeable = True

        train_data = AirlineDataset(X_train, y_train)
        test_data = AirlineDataset(X_test, y_test)

        train_loaders.append(
            DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        )
        test_loaders.append(DataLoader(test_data, batch_size=BATCH_SIZE))

    return train_loaders, test_loaders


def load_all_airports():
    train_loaders = []
    test_loaders = []

    labels = pd.DataFrame()

    data = defaultdict()

    encoder = partial(
        OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoders = defaultdict(encoder)
    df_list = []

    for airport in AIRPORTS:
        labels = pd.concat(
            [
                labels,
                pd.read_csv(
                    f"{DATA_DIR}/train_labels_phase2/phase2_train_labels_{airport}.csv.bz2",
                    parse_dates=["timestamp"],
                ),
            ]
        )

        for airline in AIRLINES:
            try:
                dfs = pd.read_csv(
                    f"{DATA_DIR}/train_tables/{airport}/{airline}_train.csv",
                    parse_dates=["timestamp"],
                    dtype={"precip": str},
                )
                dfs_val = pd.read_csv(
                    f"{DATA_DIR}/validation_tables/{airport}/{airline}_validation.csv",
                    parse_dates=["timestamp"],
                    dtype={"precip": str},
                )

            except FileNotFoundError:
                continue

            if len(dfs) == 0:
                continue

            try:
                # Modified for the submission inference script to train for full df with the full training data
                # train_dfs = dfs.iloc[train_indices]
                train_dfs = dfs
                test_dfs = dfs_val

            except ValueError:
                continue

            X_train = train_dfs[features].copy()
            y_train = train_dfs["minutes_until_pushback"].copy()
            X_test = test_dfs[features].copy()
            y_test = test_dfs["minutes_until_pushback"].copy()

            data[airline] = data.get(airline, defaultdict())
            data[airline]["X_train"] = pd.concat(
                [data[airline].get("X_train", pd.DataFrame()), X_train]
            )
            data[airline]["X_test"] = pd.concat(
                [data[airline].get("X_test", pd.DataFrame()), X_test]
            )
            data[airline]["y_train"] = pd.concat(
                [data[airline].get("y_train", pd.DataFrame()), y_train]
            )
            data[airline]["y_test"] = pd.concat(
                [data[airline].get("y_test", pd.DataFrame()), y_test]
            )
            df_list.append(dfs)
            df_list.append(dfs_val)

    full_df = pd.concat(df_list)
    for column in encoded_columns:
        try:
            encoders[column].fit(full_df[[column]])
        except Exception as e:
            print(e)
            print(column)

    for airline in data.keys():
        X_train = data[airline]["X_train"]
        X_test = data[airline]["X_test"]
        y_train = data[airline]["y_train"]
        y_test = data[airline]["y_test"]

        for column in encoded_columns:
            try:
                X_train[column] = encoders[column].transform(X_train[[column]])
                X_test[column] = encoders[column].transform(X_test[[column]])
            except Exception as e:
                print(e)
                print(column)

        for column in X_train.columns:
            if column not in encoded_columns:
                train_std = X_train[column].std()

                if train_std == 0:
                    train_std = 1

                test_std = X_test[column].std()

                if test_std == 0:
                    test_std = 1

                X_train[column] = (X_train[column] - X_train[column].mean()) / train_std
                X_test[column] = (X_test[column] - X_test[column].mean()) / test_std

        X_train = X_train.to_numpy(dtype=np.float32, copy=True)
        X_test = X_test.to_numpy(dtype=np.float32, copy=True)
        y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        X_train.flags.writeable = True
        y_train.flags.writeable = True
        X_test.flags.writeable = True
        y_test.flags.writeable = True

        train_data = AirlineDataset(X_train, y_train)
        test_data = AirlineDataset(X_test, y_test)

        train_loaders.append(
            DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        )
        test_loaders.append(DataLoader(test_data, batch_size=BATCH_SIZE))

    file = open(f"{ASSETS_DIR}/encoders.pickle", "wb")
    pickle.dump(encoders, file)
    file.close()

    return train_loaders, test_loaders
