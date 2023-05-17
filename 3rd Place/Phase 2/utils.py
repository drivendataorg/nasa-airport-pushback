from datetime import timedelta
import datetime
from typing import List, Tuple, Union
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
import pandas as pd
from loguru import logger
from config import (
    AIRLINES,
    AIRPORTS,
    AIRPORT_MASK,
    ENGINE_CLASSES,
    AIRCRAFT_TYPE,
    NORM_MIN,
    NORM_MAX,
    data_directory_raw,
    xgb_model_directory,
    private_features_directory,
    public_features_directory,
    load_bz2_file_bool,
    matt_dir_path,
)

import os
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional
import pickle
import xgboost as xgb


def get_etd_minutes_until_pushback(
    now_features: pd.DataFrame, now_etd: pd.DataFrame
) -> pd.Series:
    # get the latest ETD for each flight
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time

    # merge the latest ETD with the flights we are predicting
    departure_runway_estimated_time = now_features.merge(
        latest_now_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    minutes_until_pushback = pd.Series(
        (
            departure_runway_estimated_time
            - now_features.index.get_level_values("timestamp")
        ).dt.total_seconds()
        / 60,
        name="minutes_until_pushback",
    )
    return now_features.drop(columns=["minutes_until_pushback"]).join(
        minutes_until_pushback, how="left", on="gufi", validate="1:1"
    )[["minutes_until_pushback"]]


def count_same_aircraft_type(gufis, mfs, aircraft_type):
    return (
        mfs.loc[gufis.loc[gufis.isin(mfs.index)], "aircraft_type"] == aircraft_type
    ).sum()


def load_public_airport_features(index: pd.Index, airport: str):
    """Loads the public features for a set of flights from one airport.

    The public features are:
    - minutes_until_pushback: the latest ETD estimate for the flight
    - n_flights: the number of flights in the last 30 hours
    """
    # initialize an empty table with the desired gufi/timestamp index
    features = pd.DataFrame([], index=index)

    # Getting Airline Codes
    logger.debug("Loading Airline Code features")
    curr_airline_codes = list(index.get_level_values("gufi").str[:3])
    rows = len(curr_airline_codes)
    diff_airlines = AIRLINES
    cols = len(diff_airlines)  # TODO: Need to fix this
    data_airlines = np.zeros((rows, cols))
    # Set the corresponding values to 1
    for i, airline in enumerate(diff_airlines):
        data_airlines[:, i] = np.array(curr_airline_codes) == airline
    features[diff_airlines] = data_airlines

    # get etd features
    logger.debug("Loading ETD features")
    if matt_dir_path:
        file_path = f"{data_directory_raw}/{airport}/public/{airport}/{airport}_etd.csv"
    else:
        file_path = f"{data_directory_raw}/public/{airport}/{airport}_etd.csv"
    file_path = file_path + ".bz2" if load_bz2_file_bool else file_path
    etd = pd.read_csv(
        file_path,
        # f"{directory}{airport}/public/{airport}/{airport}_etd.csv",
        # f"{data_directory_raw} / airport / "public" / airport / f"{airport}_etd.csv",
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )

    df_train_labels = features.reset_index()
    df_train_labels["timestamp"] = pd.to_datetime(df_train_labels["timestamp"])
    print(f"df_train_labels: {len(df_train_labels)}")

    etd.sort_values(by="timestamp", inplace=True)
    etd_shorter = etd[etd.timestamp < etd.departure_runway_estimated_time]

    df_final_labels_etd = df_train_labels.copy(deep=True)
    df_final_labels_etd = df_final_labels_etd.sort_values(
        by=["gufi", "timestamp"]
    ).reset_index(drop=True)
    df_etd_limited = (
        etd_shorter[etd_shorter.gufi.isin(df_final_labels_etd.gufi.unique())]
        .sort_values(by=["gufi", "timestamp"])
        .reset_index(drop=True)
    )
    df_final_labels_etd = (
        df_final_labels_etd[df_final_labels_etd.gufi.isin(etd_shorter.gufi.unique())]
        .sort_values(by=["gufi", "timestamp"])
        .reset_index(drop=True)
    )
    unique_planes = list(df_final_labels_etd.gufi.unique())
    df_etd_limited_no_dup = df_etd_limited.drop_duplicates(subset=["gufi"])
    print(f"DF_ETD_LIMITED_NODUP: {len(df_etd_limited_no_dup)}")
    new_planes_indexes_etd = list(df_etd_limited_no_dup.index)
    dropped_fin_labels = len(df_train_labels) - len(df_final_labels_etd)
    logger.debug(
        f"There were {dropped_fin_labels} gufis from train labels because they werent in etd"
    )
    time_etd_df = pd.DataFrame(
        columns=[
            "gufi",
            "timestamp",
            "minutes_until_departure_from_timepoint",
            "minutes_until_departure_from_timestamp",
        ]
    )
    df_final_labels_etd_exp = df_final_labels_etd.copy(deep=True)
    df_final_labels_etd_exp = df_final_labels_etd_exp.drop_duplicates(subset=["gufi"])
    new_planes_indexes_labels = list(df_final_labels_etd_exp.index)

    curr_plane_counter = 1
    list_dicts_etd = []
    for index_counter, curr_train_labels_indx in enumerate(new_planes_indexes_labels):
        if len(new_planes_indexes_labels) - 1 == index_counter:
            df_etd_limited_plane = df_etd_limited.iloc[
                new_planes_indexes_etd[curr_plane_counter - 1] :
            ]
            curr_gufi = df_final_labels_etd.gufi.iloc[curr_train_labels_indx]
            list_timepoints = list(
                df_final_labels_etd.timestamp.iloc[curr_train_labels_indx:]
            )
        else:
            df_etd_limited_plane = df_etd_limited.iloc[
                new_planes_indexes_etd[curr_plane_counter - 1] : new_planes_indexes_etd[
                    curr_plane_counter
                ]
            ]
            curr_gufi = df_final_labels_etd.gufi.iloc[curr_train_labels_indx]
            list_timepoints = list(
                df_final_labels_etd.timestamp.iloc[
                    curr_train_labels_indx : new_planes_indexes_labels[
                        index_counter + 1
                    ]
                ]
            )
        dep_times_dicts = extract_etd(curr_gufi, list_timepoints, df_etd_limited_plane)
        list_dicts_etd.extend(dep_times_dicts)
        curr_plane_counter += 1
        print(
            "Done with "
            + str(index_counter + 1)
            + " out of "
            + str(len(new_planes_indexes_labels))
            + " items",
            end="\r",
        )
    time_etd_df = pd.DataFrame(list_dicts_etd)  # TODO: What to do if this is empty?
    features.reset_index(inplace=True)
    features = pd.merge(features, time_etd_df, on=["gufi", "timestamp"])
    features = features.set_index(["gufi", "timestamp"])
    features = features.drop(columns=["airport"])
    return features


def extract_etd_werrors(curr_gufi, list_timepoints, limited_etd):
    """
    Extracts and calculates various statistics related to the estimated departure times for a given gufi and timepoints.

    Args:
        curr_gufi (str): The unique identifier for the current airplane.
        list_timepoints (list): A list of timepoints (timestamps) to calculate statistics for.
        limited_etd (pd.DataFrame): A DataFrame containing limited estimated time of departure data.

    Returns:
        list: A list of dictionaries containing the calculated statistics for each timepoint.
    """

    # Initialize an empty list to store the dictionaries with calculated statistics for each timepoint
    curr_dicts = []

    # Iterate over the timepoints
    for timepoint in list_timepoints:
        try:
            # Filter the limited_etd DataFrame to find rows where the timestamp is less than the timepoint
            # and departure_runway_estimated_time is greater than the timepoint
            df_airplane_depart = limited_etd[
                (limited_etd["timestamp"] < timepoint)
                & (limited_etd["departure_runway_estimated_time"] > timepoint)
            ].reset_index(drop=True)

            # If there are any rows in the filtered DataFrame
            if len(df_airplane_depart) > 0:
                # Get the last row of the filtered DataFrame
                df_airplane_depart_last = df_airplane_depart.tail(1).reset_index(
                    drop=True
                )

                # Calculate various statistics for the filtered DataFrame
                curr_time_est_timepoints = (
                    df_airplane_depart.departure_runway_estimated_time - timepoint
                ).dt.total_seconds().values / 60
                std_time_est_timepoints = np.std(curr_time_est_timepoints)
                mean_time_est_timepoints = np.mean(curr_time_est_timepoints)
                median_time_est_timepoints = np.median(curr_time_est_timepoints)

                # Calculate the time difference between departure_runway_estimated_time and timepoint for the last row
                curr_time_est_timepoint = (
                    df_airplane_depart_last.departure_runway_estimated_time - timepoint
                ).iloc[0].total_seconds() / 60

                # Calculate the time difference between departure_runway_estimated_time and timestamp for the last row
                curr_time_est_timestamp = (
                    df_airplane_depart_last.departure_runway_estimated_time
                    - df_airplane_depart_last["timestamp"]
                ).iloc[0].total_seconds() / 60

                # Count the number of rows in the filtered DataFrame
                found_etd_vals = len(df_airplane_depart)

                # Create a dictionary with the calculated values and append it to the curr_dicts list
                ans = {
                    "gufi": curr_gufi,
                    "timestamp": timepoint,
                    "minutes_until_departure_from_timepoint": curr_time_est_timepoint,
                    "minutes_until_departure_from_timestamp": curr_time_est_timestamp,
                    "mean_departure_from_timepoint": mean_time_est_timepoints,
                    "median_departure_from_timepoint": median_time_est_timepoints,
                    "std_departure_from_timepoint": std_time_est_timepoints,
                    "found_etd_vals": found_etd_vals,
                }
                curr_dicts.append(ans)
        except:
            # If there's an exception, continue with the next timepoint
            continue

    return curr_dicts


def extract_etd(curr_gufi, list_timepoints, limited_etd):
    """
    Extracts and calculates various statistics related to the estimated departure times for a given gufi and timepoints.

    Args:
        curr_gufi (str): The unique identifier for the current airplane.
        list_timepoints (list): A list of timepoints (timestamps) to calculate statistics for.
        limited_etd (pd.DataFrame): A DataFrame containing limited estimated time of departure data.

    Returns:
        list: A list of dictionaries containing the calculated statistics for each timepoint.
    """

    # Initialize an empty list to store the dictionaries with calculated statistics for each timepoint
    curr_dicts = []

    # Iterate over the timepoints
    for timepoint in list_timepoints:
        try:
            # Filter the limited_etd DataFrame to find rows where the timestamp is less than the timepoint
            # and departure_runway_estimated_time is greater than the timepoint
            df_airplane_depart = limited_etd[
                (limited_etd["timestamp"] < timepoint)
                & (limited_etd["departure_runway_estimated_time"] > timepoint)
            ].reset_index(drop=True)

            # If there are any rows in the filtered DataFrame
            if len(df_airplane_depart) > 0:
                # Get the last row of the filtered DataFrame
                df_airplane_depart_last = df_airplane_depart.tail(1).reset_index(
                    drop=True
                )

                # Calculate various statistics for the filtered DataFrame
                curr_time_est_timepoints = (
                    df_airplane_depart.departure_runway_estimated_time - timepoint
                ).dt.total_seconds().values / 60
                std_time_est_timepoints = np.std(curr_time_est_timepoints)
                mean_time_est_timepoints = np.mean(curr_time_est_timepoints)
                median_time_est_timepoints = np.median(curr_time_est_timepoints)

                # Calculate the time difference between departure_runway_estimated_time and timepoint for the last row
                curr_time_est_timepoint = (
                    df_airplane_depart_last.departure_runway_estimated_time - timepoint
                ).iloc[0].total_seconds() / 60

                # Calculate the time difference between departure_runway_estimated_time and timestamp for the last row
                curr_time_est_timestamp = (
                    df_airplane_depart_last.departure_runway_estimated_time
                    - df_airplane_depart_last["timestamp"]
                ).iloc[0].total_seconds() / 60

                # Count the number of rows in the filtered DataFrame
                found_etd_vals = len(df_airplane_depart)

                # Create a dictionary with the calculated values and append it to the curr_dicts list
                ans = {
                    "gufi": curr_gufi,
                    "timestamp": timepoint,
                    "minutes_until_departure_from_timepoint": curr_time_est_timepoint,
                    "minutes_until_departure_from_timestamp": curr_time_est_timestamp,
                    "mean_departure_from_timepoint": mean_time_est_timepoints,
                    "median_departure_from_timepoint": median_time_est_timepoints,
                    "std_departure_from_timepoint": std_time_est_timepoints,
                    "found_etd_vals": found_etd_vals,
                }
                curr_dicts.append(ans)
            else:
                # If there are no rows in the filtered DataFrame, append a dictionary with NaN values to the curr_dicts list
                ans = {
                    "gufi": curr_gufi,
                    "timestamp": timepoint,
                    "minutes_until_departure_from_timepoint": 0,
                    "minutes_until_departure_from_timestamp": 0,
                    "mean_departure_from_timepoint": 0,
                    "median_departure_from_timepoint": 0,
                    "std_departure_from_timepoint": 0,
                    "found_etd_vals": 0,
                }
                curr_dicts.append(ans)
        except:
            # If there's an exception, continue with the next timepoint
            ans = {
                "gufi": curr_gufi,
                "timestamp": timepoint,
                "minutes_until_departure_from_timepoint": np.nan,
                "minutes_until_departure_from_timestamp": np.nan,
                "mean_departure_from_timepoint": np.nan,
                "median_departure_from_timepoint": np.nan,
                "std_departure_from_timepoint": np.nan,
                "found_etd_vals": 0,
            }
            curr_dicts.append(ans)
            continue

    return curr_dicts


def load_private_airline_airport_features(index: pd.Index, airline: str, airport: str):
    """Loads the private features for a set of flights from one airline and one airport."""
    features = pd.DataFrame([], index=index)

    if matt_dir_path:
        file_path = f"{data_directory_raw}/{airport}/private/{airport}/{airport}_{airline}_mfs.csv"
    else:
        file_path = (
            f"{data_directory_raw}/private/{airport}/{airport}_{airline}_mfs.csv"
        )
    file_path = file_path + ".bz2" if load_bz2_file_bool else file_path
    mfs = (
        pd.read_csv(
            # f"{directory}{airport}/private/{airport}/{airport}_{airline}_mfs.csv",
            file_path,  # ------------------------- FILE PATH DISCREPANCY -------------
            usecols=["gufi", "aircraft_engine_class", "aircraft_type"],
        )
        .drop_duplicates(subset=["gufi"], keep="first")
        .set_index("gufi")
    )

    features = features.merge(
        mfs, how="left", left_index=True, right_index=True, validate="1:1"
    )

    # Loading in ETD
    if matt_dir_path:
        file_path = f"{data_directory_raw}/{airport}/public/{airport}/{airport}_etd.csv"
    else:
        file_path = f"{data_directory_raw}/public/{airport}/{airport}_etd.csv"
    file_path = file_path + ".bz2" if load_bz2_file_bool else file_path
    etd = pd.read_csv(
        # f"{directory}{airport}/public/{airport}/{airport}_etd.csv",
        file_path,
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )
    etd_airline = etd[etd["gufi"].str.contains(airline)]

    if matt_dir_path:
        file_path = f"{data_directory_raw}/{airport}/private/{airport}/{airport}_{airline}_standtimes.csv"
    else:
        file_path = (
            f"{data_directory_raw}/private/{airport}/{airport}_{airline}_standtimes.csv"
        )
    file_path = file_path + ".bz2" if load_bz2_file_bool else file_path
    # Loading in Standtimes
    standtimes = pd.read_csv(
        # f"{directory}{airport}/private/{airport}/{airport}_{airline}_standtimes.csv",
        file_path,
        parse_dates=[
            "timestamp",
            "arrival_stand_actual_time",
            "departure_stand_actual_time",
        ],
    )
    standtimes[
        ["plane_id", "departing_airport_code", "arriving_airport_code"]
    ] = standtimes.gufi.str.split(".", expand=True)[[0, 1, 2]]
    standtimes["airline_code"] = standtimes.gufi.str[:3]

    timestamps = pd.to_datetime(features.index.get_level_values("timestamp"))

    # We will keep track of the feature updates in a dictionary
    # The dictionary keys are timestamps and the values are lists of tuples. Each tuple contains a gufi and the corresponding aircraft count
    feature_updates = defaultdict(list)

    for i, now in enumerate(timestamps):

        # Getting ETD Features
        now_features = features.xs(now, level="timestamp", drop_level=False)
        now_etd = etd_airline.loc[
            (etd_airline.timestamp > now - timedelta(hours=30))
            & (etd_airline.timestamp <= now)
        ]
        now_gufis = pd.Series(now_etd.gufi.unique())
        # Get the aircraft type for this row
        aircraft_type = features.iloc[i]["aircraft_type"]
        aircraft_count = count_same_aircraft_type(now_gufis, mfs, aircraft_type)

        # Getting stand times features
        destination = features.index[i][0].split(".")[2]
        standtimes_dest = standtimes[
            (standtimes.timestamp > now - timedelta(hours=30))
            & (standtimes.timestamp <= now)
        ]
        standtimes_dest = standtimes_dest[
            standtimes_dest["arriving_airport_code"] == destination
        ]
        times_to_dest = (
            standtimes_dest["arrival_stand_actual_time"]
            - standtimes_dest["departure_stand_actual_time"]
        )
        mean_time_to_dest = times_to_dest.mean().total_seconds() / 60
        std_time_to_dest = times_to_dest.std().total_seconds() / 60
        median_time_to_dest = times_to_dest.median().total_seconds() / 60

        feature_updates[now].append(
            (
                features.index[i],
                aircraft_count,
                mean_time_to_dest,
                std_time_to_dest,
                median_time_to_dest,
            )
        )

        # print("Done with " + str(i+1) + " out of " + str(features.shape[0]) + " items", end='\r')

    # Convert the feature_updates dictionary to a DataFrame
    feature_updates_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "gufi": gufi[0],
                "n_same_aircraft_type": aircraft_count,
                "mean_time_to_dest": mean_time_to_dest,
                "std_time_to_dest": std_time_to_dest,
                "median_time_to_dest": median_time_to_dest,
            }
            for timestamp, updates in feature_updates.items()
            for gufi, aircraft_count, mean_time_to_dest, std_time_to_dest, median_time_to_dest in updates
        ]
    ).set_index(["timestamp", "gufi"])

    # Join features with the
    features = features.merge(
        feature_updates_df, how="left", left_index=True, right_index=True
    )

    return features


def one_hot_encode(feature_df):
    #'airport'
    #'aircraft_engine_class'
    #'aircraft_type'

    ap_encoded_df = pd.get_dummies(
        feature_df["airport"], prefix="", prefix_sep=""
    ).reindex(columns=AIRPORTS, fill_value=0)
    ec_encoded_df = pd.get_dummies(
        feature_df["aircraft_engine_class"], prefix="", prefix_sep=""
    ).reindex(columns=ENGINE_CLASSES, fill_value=0)
    at_encoded_df = pd.get_dummies(
        feature_df["aircraft_type"], prefix="", prefix_sep=""
    ).reindex(columns=AIRCRAFT_TYPE, fill_value=0)

    new_df = pd.concat(
        [feature_df, ap_encoded_df, ec_encoded_df, at_encoded_df], axis=1
    )
    del new_df["airport"]
    del new_df["aircraft_engine_class"]
    del new_df["aircraft_type"]

    return new_df


def convert_datetime(feature_df):
    feature_df["month"] = feature_df["timestamp"].dt.month
    feature_df["day"] = feature_df["timestamp"].dt.day
    feature_df["hour"] = feature_df["timestamp"].dt.hour
    feature_df["minute"] = feature_df["timestamp"].dt.minute

    del feature_df["timestamp"]

    return feature_df


def enforce_type(feature_df, type=np.float64):
    for col in feature_df.columns:
        if feature_df[col].dtype != type:
            feature_df[col] = feature_df[col].astype(type)

    return feature_df


def norm_data(feature_df):
    for col in feature_df.columns:
        feature_df[col] = (feature_df[col] - NORM_MIN[col]) / (
            NORM_MAX[col] - NORM_MIN[col]
        )
    return feature_df


def private_data_model_specific_preprocessing(feature_df, label_df, type):
    del feature_df["gufi"]
    feature_df = one_hot_encode(feature_df)
    feature_df = convert_datetime(feature_df)
    feature_df = enforce_type(feature_df, type=type)
    feature_df = norm_data(feature_df)
    feature_df = feature_df.fillna(0)

    label_df = label_df["minutes_until_pushback"]
    label_df.astype(type)

    return feature_df, label_df


def check_airline_saved(airline: str):
    features_fp = f"{private_features_directory}/{airline}_features.csv"
    labels_fp = f"{private_features_directory}/{airline}_labels.csv"
    return os.path.isfile(features_fp) and os.path.isfile(labels_fp)


def get_airline_labels(labels: pd.DataFrame, airline: str) -> pd.DataFrame:
    logger.debug(f"Loading airline labels for {airline}")
    airlines = pd.Series(
        labels.index.get_level_values("gufi").str[:3], index=labels.index
    )
    return labels.loc[airlines == airline]


def load_airport_labels(airport: str, downsample: Optional[int] = 100_000):
    if downsample:

        def skiprows(i):
            if i == 0:
                return False
            return i % downsample

    else:
        skiprows = None

    logger.debug(f"Loading labels for {airport}")
    file_path = f"{data_directory_raw}/phase2_train_labels_{airport}.csv"
    file_path = file_path + ".bz2" if load_bz2_file_bool else file_path

    return pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
        skiprows=skiprows,
        index_col=["gufi", "timestamp", "airport"],
    )


def train_public_model():
    presentDate = datetime.datetime.now()
    save_dir_name = str(int(presentDate.timestamp()))
    save_path = f"{xgb_model_directory}/{save_dir_name}"
    os.mkdir(save_path)
    for airport in AIRPORTS:
        if AIRPORT_MASK[airport]:
            fn = f"{public_features_directory}/processed_public_features_{airport}.pkl"
            print(airport)
            with open(fn, "rb") as f:
                # Load the object from the pickle file
                curr_data = pickle.load(f)
            curr_data["X"].index
            train_data = curr_data["X"].reset_index()
            # extract year, month, day, and hour information
            train_data["unix_time"] = (
                train_data["timestamp"].astype(np.int64) // 10 ** 9
            )
            train_data.dropna(inplace=True)
            train_data.reset_index(drop=True, inplace=True)
            label_data = curr_data["y"]
            merged_df = pd.merge(
                train_data, label_data, how="left", on=["timestamp", "gufi"]
            )
            feature_cols = merged_df.columns
            feature_cols = list(
                set(feature_cols) - set(["gufi", "minutes_until_pushback", "timestamp"])
            )
            label_col = ["minutes_until_pushback"]
            gss = GroupShuffleSplit(
                n_splits=1, train_size=0.7, test_size=0.3, random_state=42
            )
            # split the data into train and test sets
            # train_index, test_index = next(gss.split(merged_df, groups=merged_df['gufi']))
            # df_train = merged_df.iloc[train_index].copy()
            # df_test = merged_df.iloc[test_index].copy()
            df_train = merged_df.copy()
            X_internal_train = df_train[list(set(feature_cols))]
            y_internal_train = df_train[label_col]
            # X_internal_test = df_test[list(set(feature_cols))]
            # y_internal_test = df_test[label_col]
            internal_regressor = xgb.XGBRegressor()
            print(f"Internal Regressor Len of Training: {len(X_internal_train)}")
            internal_regressor.fit(X_internal_train, y_internal_train)
            # print(f"Internal Regressor Len of Prediction: {len(X_internal_test)}")
            # y_internal_pred = np.int32(np.around(internal_regressor.predict(X_internal_test),decimals=0))
            # internal_mae = mean_absolute_error(y_internal_test, y_internal_pred)
            # internal_mse = mean_squared_error(y_internal_test, y_internal_pred)
            # # Print the results
            # print(f"MAE: {internal_mae} & MSE: {internal_mse}")
            # df_test['y_pred'] = y_internal_pred
            model_file_name = f"{save_path}/{airport}_xgb.pkl"
            curr_model = {}
            curr_model["features"] = feature_cols
            curr_model["model"] = internal_regressor
            pickle.dump(curr_model, open(model_file_name, "wb"))


def initialize_model(n_features: int = 122):
    """
    Testing simple model to start for integration. Once there is a better understanding with
    others about features and where loading happens can apply changes.
    """
    logger.debug(f"MODEL HAS BEEN initialized")
    input_shape = (n_features,)
    input_layer = layers.Input(shape=input_shape)
    batch_norm = layers.BatchNormalization()(input_layer)
    dense0 = layers.Dense(n_features * 2, activation="relu")(batch_norm)
    dense1 = layers.Dense(n_features * 2, activation="relu")(dense0)
    dense2 = layers.Dense(n_features * 2, activation="relu")(dense1)
    output_layer = layers.Dense(1, activation="relu")(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def load_public_model(model_dir, airport_lists):
    loaded_models = {}
    for curr_airport_name in airport_lists:
        with open(f"{model_dir}/{curr_airport_name}_xgb.pkl", "rb") as f:
            loaded_models[curr_airport_name] = pickle.load(f)
    return loaded_models


class ProcessedPublicFeatures:
    public_features_dict = {}

    def set_public_features(self, dict_with_values):
        self.public_features_dict = dict_with_values

    def get_public_features(self):
        return self.public_features_dict
