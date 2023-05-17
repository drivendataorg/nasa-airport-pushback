import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_etdv3(curr_gufi, list_timepoints, limited_etd):
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
    Extracts and calculates various statistics related to the estimated departure times for a given gufi and timepoints
    within a 30-hour window.

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
            # and within the last 30 hours
            df_airplane_depart = limited_etd[
                (limited_etd["timestamp"] < timepoint)
                & (limited_etd["timestamp"] > timepoint - np.timedelta64(30, "h"))
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

                # Calculate the time difference between departure_runway_estimated_time and timepoint for the last row
                curr_time_est_timepoint = (
                    df_airplane_depart_last.departure_runway_estimated_time - timepoint
                ).iloc[0].total_seconds() / 60

                # Calculate the time difference between departure_runway_estimated_time and timestamp for the last row
                curr_time_est_timestamp = (
                    df_airplane_depart_last.departure_runway_estimated_time
                    - df_airplane_depart_last["timestamp"]
                ).iloc[0].total_seconds() / 60

                # Create a dictionary with the calculated values and append it to the curr_dicts list
                ans = {
                    "gufi": curr_gufi,
                    "timestamp": timepoint,
                    "minutes_until_departure_from_timepoint": curr_time_est_timepoint,
                    "minutes_until_departure_from_timestamp": curr_time_est_timestamp,
                    "mean_departure_from_timepoint": mean_time_est_timepoints,
                    "std_departure_from_timepoint": std_time_est_timepoints,
                }
                curr_dicts.append(ans)
        except:
            # If there's an exception, continue with the next timepoint
            continue

    return curr_dicts


def extract_combined_vals_taxi_error_estpush_v2(
    timepoint, df_stand_depart_etd, lookback_time=120
):
    """
    Extracts and calculates various statistics related to taxi times, errors, and pushback times within a specified
    lookback window.

    Args:
        timepoint (datetime): The timestamp to consider for calculations.
        df_stand_depart_etd (pd.DataFrame): A DataFrame containing stand departure and estimated time of departure data.
        lookback_time (int, optional): The lookback window in minutes. Default is 120 minutes.

    Returns:
        list: A list containing the calculated mean and standard deviation values for taxi times, errors, and pushback times.
    """

    # Filter the DataFrame based on the given timepoint and lookback_time
    df_temp = df_stand_depart_etd[
        (df_stand_depart_etd.timestamp_standtimes < timepoint)
        & (
            df_stand_depart_etd.timestamp_standtimes
            >= timepoint - np.timedelta64(lookback_time, "m")
        )
    ].reset_index(drop=True)

    # While loop to ensure at least 4 records are found or until lookback_time reaches 180 minutes
    while len(df_temp) < 4 and lookback_time < 180:
        lookback_time += 60
        df_temp = df_stand_depart_etd[
            (df_stand_depart_etd.timestamp_standtimes < timepoint)
            & (
                df_stand_depart_etd.timestamp_standtimes
                >= timepoint - np.timedelta64(lookback_time, "m")
            )
        ].reset_index(drop=True)

    # Calculate various statistics for the filtered DataFrame
    found_counts = len(df_temp)
    taxi_times_mean = df_temp.taxi_time.mean()
    taxi_times_std = df_temp.taxi_time.std()
    error_etd_mean = df_temp.etd_error.mean()
    error_etd_std = df_temp.etd_error.std()
    pushback_mean = df_temp.prev_min_until_pushback.mean()
    pushback_std = df_temp.prev_min_until_pushback.std()
    pushback_error_etd_mean = df_temp.prev_min_until_pushback_error.mean()
    pushback_error_etd_std = df_temp.prev_min_until_pushback_error.std()

    return [
        taxi_times_mean,
        taxi_times_std,
        error_etd_mean,
        error_etd_std,
        pushback_mean,
        pushback_std,
        pushback_error_etd_mean,
        pushback_error_etd_std,
        found_counts,
    ]


def extract_combined_vals_taxi_error_estpush(timepoint, df_stand_depart_etd):
    """
    Extracts and calculates various statistics related to taxi times, errors, and pushback times for a given timepoint
    and DataFrame.

    Args:
        timepoint (datetime): The timestamp to consider for calculations.
        df_stand_depart_etd (pd.DataFrame): A DataFrame containing stand departure and estimated time of departure data.

    Returns:
        list: A list containing the calculated mean and standard deviation values for taxi times, errors, and pushback times.
    """

    # Filter the DataFrame based on the given timepoint
    df_temp = df_stand_depart_etd[
        (df_stand_depart_etd.timestamp_standtimes < timepoint)
    ].reset_index(drop=True)

    # Calculate taxi times and their statistics
    df_taxi_times = (
        df_temp.departure_runway_actual_time - df_temp.departure_stand_actual_time
    ).dt.total_seconds().values / 60
    taxi_times_mean = np.mean(df_taxi_times)
    taxi_times_std = np.std(df_taxi_times)

    # Calculate error between estimated and actual departure time
    error_etd_df = (
        np.abs(
            df_temp.departure_runway_actual_time
            - df_temp.departure_runway_estimated_time
        )
        .dt.total_seconds()
        .values
        / 60
    )
    error_etd_mean = np.mean(error_etd_df)
    error_etd_std = np.std(error_etd_df)

    # Calculate pushback times and their statistics
    df_pushback_etd = (
        (df_temp.departure_runway_estimated_time - df_temp.timestamp_etd)
        - (df_temp.departure_runway_actual_time - df_temp.departure_stand_actual_time)
    ).dt.total_seconds().values / 60
    pushback_mean = np.mean(df_pushback_etd)
    pushback_std = np.std(df_pushback_etd)

    # Calculate pushback error
    df_pushback_error = np.abs(
        (
            df_temp.departure_stand_actual_time
            - (
                (
                    (df_temp.departure_runway_estimated_time - df_temp.timestamp_etd)
                    - (
                        df_temp.departure_runway_actual_time
                        - df_temp.departure_stand_actual_time
                    )
                )
                + df_temp.timestamp_etd
            )
        )
        .dt.total_seconds()
        .values
        / 60
    )
    pushback_error_etd_mean = np.mean(error_etd_df)
    pushback_error_etd_std = np.std(error_etd_df)

    return [
        taxi_times_mean,
        taxi_times_std,
        error_etd_mean,
        error_etd_std,
        pushback_mean,
        pushback_std,
        pushback_error_etd_mean,
        pushback_error_etd_std,
    ]


def data_loader_submission_train_labels(directory, airport):
    """
    Loads the submission train labels data from a specified directory and filters it based on the given airport.

    Args:
        directory (str): The directory containing the submission_data.csv file.
        airport (str): The airport code to filter the data by.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered submission train labels data.
    """

    # Load the submission_data.csv file from the specified directory
    df_train_labels = pd.read_csv(f"{directory}submission_data.csv")
    print(f"Loading from: {directory}submission_data.csv")

    # Convert timestamp to pandas datetime
    df_train_labels["timestamp"] = pd.to_datetime(df_train_labels["timestamp"])

    # Filter the DataFrame based on the given airport
    df_train_labels = df_train_labels[df_train_labels.airport == airport]

    return df_train_labels


def data_loader_submission_etd(directory, airport):
    """
    Loads the submission Estimated Time of Departure (ETD) data from a specified directory and filters it based on the given airport.

    Args:
        directory (str): The directory containing the ETD data files.
        airport (str): The airport code to filter the data by.

    Returns:
        pd.DataFrame: A tuple containing the filtered submission train labels data and the filtered ETD data.
    """

    # Load submission train labels data for the specified airport
    df_train_labels = data_loader_submission_train_labels(directory, airport)

    # Load the ETD data for the specified airport
    df_etd = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_etd.csv")

    # Convert timestamps to pandas datetime
    df_etd["timestamp"] = pd.to_datetime(df_etd["timestamp"])
    df_etd["departure_runway_estimated_time"] = pd.to_datetime(
        df_etd["departure_runway_estimated_time"]
    )

    # Sort the DataFrame by the timestamp column
    df_etd.sort_values(by="timestamp", inplace=True)

    # Filter the DataFrame to keep only rows with timestamp less than departure_runway_estimated_time
    df_etd_shorter = df_etd[df_etd.timestamp < df_etd.departure_runway_estimated_time]

    return df_train_labels, df_etd_shorter


def data_loader_train_labels(directory, airport):
    """
    Loads the train labels data for a specified airport from a given directory.

    Args:
        directory (str): The directory containing the train labels data files.
        airport (str): The airport code to load the data for.

    Returns:
        pd.DataFrame: The loaded train labels DataFrame for the specified airport.
    """

    # Load train labels data for the specified airport
    df_train_labels = pd.read_csv(f"{directory}{airport}/train_labels_{airport}.csv")

    # Convert timestamp to pandas datetime
    df_train_labels["timestamp"] = pd.to_datetime(df_train_labels["timestamp"])

    return df_train_labels


def data_loader_metadata(directory, airport):
    """
    Loads the train labels and metadata for a specified airport from a given directory.

    Args:
        directory (str): The directory containing the train labels and metadata files.
        airport (str): The airport code to load the data for.

    Returns:
        pd.DataFrame: The loaded train labels DataFrame for the specified airport.
        pd.DataFrame: The transformed metadata DataFrame for the specified airport.
    """

    # Load train labels data for the specified airport
    df_train_labels = pd.read_csv(f"{directory}{airport}/train_labels_{airport}.csv")

    # Convert timestamp to pandas datetime
    df_train_labels["timestamp"] = pd.to_datetime(df_train_labels["timestamp"])

    # Load metadata for the specified airport
    df_mfs = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_mfs.csv")

    # Fill missing values with 'Unknown'
    df_mfs.fillna("Unknown", inplace=True)

    # Convert 'isdeparture' column to integer
    df_mfs["isdeparture"] = df_mfs["isdeparture"].astype(int)

    # Perform one-hot encoding for specific columns with specified prefixes
    df_one_hot = pd.get_dummies(
        df_mfs[
            ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]
        ],
        prefix=[
            "aircraft_engine_class",
            "aircraft_type",
            "major_carrier",
            "flight_type",
        ],
    )

    # Join the one-hot encoded columns to the original DataFrame
    df_mfs = df_mfs.join(df_one_hot)

    # Keep only columns with specified prefixes and 'gufi'
    to_keep_prefixes = ["aircraft_engine_class", "major_carrier", "flight_type", "gufi"]
    df_mfs_keep = df_mfs.loc[:, df_mfs.columns.str.startswith(tuple(to_keep_prefixes))]

    # Drop unnecessary columns
    df_mfs_keep.drop(
        columns=["aircraft_engine_class", "major_carrier", "flight_type"], inplace=True
    )

    return df_train_labels, df_mfs_keep


def data_loader_etd(directory, airport):
    """
    Loads the train labels and estimated time of departure (ETD) data for a specified airport from a given directory.

    Args:
        directory (str): The directory containing the train labels and ETD files.
        airport (str): The airport code to load the data for.

    Returns:
        pd.DataFrame: The loaded train labels DataFrame for the specified airport.
        pd.DataFrame: The filtered ETD DataFrame for the specified airport.
    """

    # Load train labels data for the specified airport
    df_train_labels = pd.read_csv(f"{directory}{airport}/train_labels_{airport}.csv")

    # Convert timestamp to pandas datetime
    df_train_labels["timestamp"] = pd.to_datetime(df_train_labels["timestamp"])

    # Load ETD data for the specified airport
    df_etd = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_etd.csv")

    # Convert timestamps to pandas datetime
    df_etd["timestamp"] = pd.to_datetime(df_etd["timestamp"])
    df_etd["departure_runway_estimated_time"] = pd.to_datetime(
        df_etd["departure_runway_estimated_time"]
    )

    # Sort DataFrame by timestamp column
    df_etd.sort_values(by="timestamp", inplace=True)

    # Filter ETD DataFrame to keep only rows where timestamp is less than departure_runway_estimated_time
    df_etd_shorter = df_etd[df_etd.timestamp < df_etd.departure_runway_estimated_time]

    return df_train_labels, df_etd_shorter


def data_loader(directory, airport):
    """
    Loads various datasets for a specified airport from a given directory.

    Args:
        directory (str): The directory containing the datasets.
        airport (str): The airport code to load the data for.

    Returns:
        pd.DataFrame: The loaded config DataFrame for the specified airport.
        pd.DataFrame: The loaded ETD DataFrame for the specified airport.
        pd.DataFrame: The loaded first_position DataFrame for the specified airport.
        pd.DataFrame: The loaded LAMP DataFrame for the specified airport.
        pd.DataFrame: The loaded MFS DataFrame for the specified airport.
        pd.DataFrame: The loaded runway_arrival DataFrame for the specified airport.
        pd.DataFrame: The loaded runway_departure DataFrame for the specified airport.
        pd.DataFrame: The loaded standtimes DataFrame for the specified airport.
    """

    # Load config data for the specified airport
    df_config = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_config.csv")
    df_config["departure_runways"].fillna("Unknown", inplace=True)
    df_config["arrival_runways"].fillna("Unknown", inplace=True)

    # Convert timestamps to pandas datetime
    df_config["timestamp"] = pd.to_datetime(df_config["timestamp"])
    df_config["start_time"] = pd.to_datetime(df_config["start_time"])

    # sorting by time column
    df_config.sort_values(by="timestamp", inplace=True)
    ## One Hot Encoding Runways
    # Split the 'departure_runways' column into a list of runway codes
    df_config["departure_runways"] = df_config["departure_runways"].str.split(", ")
    # Create a new column for the number of runway codes
    df_config["num_departure_runways"] = [
        len(row) if row[0] != "Unknown" else 0 for row in df_config["departure_runways"]
    ]
    # Get a list of unique runway codes
    runways = list(
        set(
            [
                item
                for sublist in df_config["departure_runways"].tolist()
                for item in sublist
            ]
        )
    )
    # Make the column names
    runways = [f"{r}_dep" for r in runways]
    # Create new columns for each runway code
    for runway in runways:
        df_config[runway] = [
            1 if runway[: runway.find("_")] in row else 0
            for row in df_config["departure_runways"]
        ]

    # Split the 'arrival_runways' column into a list of runway codes
    df_config["arrival_runways"] = df_config["arrival_runways"].str.split(", ")
    # Create a new column for the number of runway codes
    df_config["num_arrival_runways"] = [
        len(row) if row[0] != "Unknown" else 0 for row in df_config["arrival_runways"]
    ]
    # Get a list of unique runway codes
    runways = list(
        set(
            [
                item
                for sublist in df_config["arrival_runways"].tolist()
                for item in sublist
            ]
        )
    )
    # Make the column names
    runways = [f"{r}_arr" for r in runways]
    # Create new columns for each runway code
    for runway in runways:
        df_config[runway] = [
            1 if runway[: runway.find("_")] in row else 0
            for row in df_config["arrival_runways"]
        ]

    # Drop original departure_runways and arrival_runways columns
    df_config.drop(columns=["departure_runways", "arrival_runways"], inplace=True)

    ######ETD
    df_etd = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_etd.csv")

    # Convert timestamps to pandas datetime
    df_etd["timestamp"] = pd.to_datetime(df_etd["timestamp"])

    df_etd["departure_runway_estimated_time"] = pd.to_datetime(
        df_etd["departure_runway_estimated_time"]
    )

    # sorting by time column
    df_etd.sort_values(by="timestamp", inplace=True)

    ####first position
    df_first_pos = pd.read_csv(
        f"{directory}{airport}/{airport}/{airport}_first_position.csv"
    )

    # Convert timestamps to pandas datetime
    df_first_pos["timestamp"] = pd.to_datetime(df_first_pos["timestamp"])

    #####LAMP
    df_lamp = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_lamp.csv")

    # Convert timestamps to pandas datetime
    df_lamp["timestamp"] = pd.to_datetime(df_lamp["timestamp"])
    df_lamp["forecast_timestamp"] = pd.to_datetime(df_lamp["forecast_timestamp"])

    # sorting by time column
    df_lamp.sort_values(by="timestamp", inplace=True)
    df_lamp["precip"] = df_lamp["precip"].astype(int)

    # Integer encoding lightning prob
    lightning_prob_mapping = {"N": 0, "L": 1, "M": 2, "H": 3}
    df_lamp["lightning_prob"] = df_lamp["lightning_prob"].map(lightning_prob_mapping)

    # One hot encoding cloud
    df_one_hot = pd.get_dummies(df_lamp["cloud"])
    df_lamp = df_lamp.join(df_one_hot)
    df_lamp.drop(columns="cloud", inplace=True)

    ###MFS
    df_mfs = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_mfs.csv")
    df_mfs.fillna("Unknown", inplace=True)
    df_mfs["isdeparture"] = df_mfs["isdeparture"].astype(int)

    # one-hot encoding with specified prefixes
    df_one_hot = pd.get_dummies(
        df_mfs[
            ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]
        ],
        prefix=[
            "aircraft_engine_class",
            "aircraft_type",
            "major_carrier",
            "flight_type",
        ],
    )
    # join the one-hot encoded columns to the original DataFrame
    df_mfs = df_mfs.join(df_one_hot)

    df_mfs.drop(
        columns=[
            "aircraft_engine_class",
            "aircraft_type",
            "major_carrier",
            "flight_type",
        ],
        inplace=True,
    )

    #####runways
    df_runways = pd.read_csv(f"{directory}{airport}/{airport}/{airport}_runways.csv")

    # Convert timestamps to pandas datetime
    df_runways["timestamp"] = pd.to_datetime(df_runways["timestamp"])
    df_runways["departure_runway_actual_time"] = pd.to_datetime(
        df_runways["departure_runway_actual_time"]
    )

    # sorting by time column
    df_runways.sort_values(by="timestamp", inplace=True)

    # Need to make two seperate dfs for departure and arrival
    df_runway_departure = df_runways[df_runways["arrival_runway_actual"].isna()]
    df_runway_departure.drop(
        columns=["arrival_runway_actual", "arrival_runway_actual_time"], inplace=True
    )

    df_runway_arrival = df_runways[df_runways["departure_runway_actual"].isna()]
    df_runway_arrival.drop(
        columns=["departure_runway_actual", "departure_runway_actual_time"],
        inplace=True,
    )

    ####standtimes
    df_standtimes = pd.read_csv(
        f"{directory}{airport}/{airport}/{airport}_standtimes.csv"
    )
    df_standtimes.sort_values(by="timestamp", inplace=True)

    # Convert timestamps to pandas datetime
    df_standtimes["timestamp"] = pd.to_datetime(df_standtimes["timestamp"])
    df_standtimes["arrival_stand_actual_time"] = pd.to_datetime(
        df_standtimes["arrival_stand_actual_time"]
    )
    df_standtimes["departure_stand_actual_time"] = pd.to_datetime(
        df_standtimes["departure_stand_actual_time"]
    )

    return (
        df_config,
        df_etd,
        df_first_pos,
        df_lamp,
        df_mfs,
        df_runway_arrival,
        df_runway_departure,
        df_standtimes,
    )


def extract_exp_math_2(curr_timepoint, etd_eq_dfs, df_etd_shortened):
    """
    Extracts statistics of interest from the given DataFrames based on a given timepoint.

    Args:
        curr_timepoint (pd.Timestamp): The current timepoint to extract the statistics for.
        etd_eq_dfs (pd.DataFrame): The DataFrame containing information about ETD and standtimes.
        df_etd_shortened (pd.DataFrame): The DataFrame containing shortened ETD information.

    Returns:
        dict: A dictionary containing the extracted statistics.
    """

    # Initialize variables
    unique_cols = set()
    list_dict = []

    # Iterate through rows in etd_eq_dfs
    for curr_row in etd_eq_dfs.iterrows():
        curr_gufi = curr_row[1].gufi
        curr_actual_standtime = curr_row[1].departure_stand_actual_time
        curr_actual_departure = curr_row[1].departure_runway_actual_time
        curr_gufi_df_etd_shortened = df_etd_shortened[
            (df_etd_shortened.gufi == curr_gufi)
            & (df_etd_shortened.timestamp < curr_actual_standtime)
        ]
        bucket_tuples = [(0, 60), (60, 120), (120, 500)]
        found_dict = {}
        found_dict["gufi"] = curr_gufi
        actual_taxitime = curr_actual_departure - curr_actual_standtime
        found_dict["actual_taxitime"] = np.abs(actual_taxitime.total_seconds())
        for index, curr_tuple in enumerate(bucket_tuples):
            curr_min_lower_interval = curr_tuple[1]
            curr_min_upper_interval = curr_tuple[0]
            etd_error_average = None
            pred_taxitime_average = None
            key_format_base = f"{curr_min_upper_interval}_{curr_min_lower_interval}"
            curr_gufi_df_etd_shortened_2 = curr_gufi_df_etd_shortened[
                (
                    curr_gufi_df_etd_shortened.timestamp
                    >= curr_actual_standtime
                    - np.timedelta64(curr_min_lower_interval, "m")
                )
                & (
                    curr_gufi_df_etd_shortened.timestamp
                    <= curr_actual_standtime
                    - np.timedelta64(curr_min_upper_interval, "m")
                )
            ]
            etd_error_average = np.abs(
                (
                    curr_gufi_df_etd_shortened_2.departure_runway_estimated_time
                    - curr_actual_departure
                )
                .mean(numeric_only=False)
                .total_seconds()
            )
            pred_taxitime_average = np.abs(
                (
                    curr_gufi_df_etd_shortened_2.departure_runway_estimated_time
                    - curr_actual_standtime
                )
                .mean(numeric_only=False)
                .total_seconds()
            )
            if len(curr_gufi_df_etd_shortened_2) > 0:
                found_dict[f"len_{key_format_base}"] = len(curr_gufi_df_etd_shortened_2)
            else:
                found_dict[f"len_{key_format_base}"] = None
            unique_cols.add(f"len_{key_format_base}")
            found_dict[f"etd_error_average_{key_format_base}"] = etd_error_average
            unique_cols.add(f"etd_error_average_{key_format_base}")
            found_dict[
                f"pred_taxitime_average_{key_format_base}"
            ] = pred_taxitime_average
            unique_cols.add(f"pred_taxitime_average_{key_format_base}")
        list_dict.append(found_dict)

    # Create a new DataFrame with the list of dictionaries
    new_df = pd.DataFrame(list_dict)
    etd_eq_dfs = etd_eq_dfs.merge(new_df, how="left", on="gufi")

    # Initialize the final dictionary
    fin_dict = {}
    fin_dict["timestamp"] = curr_timepoint
    etd_eq_dfs_mean = etd_eq_dfs.describe().loc["mean"]
    etd_eq_dfs_std = etd_eq_dfs.describe().loc["std"]
    etd_eq_dfs_count = etd_eq_dfs.describe().loc["count"]

    # Extract average_actual_taxitime and std_actual_taxitime
    average_actual_taxitime = etd_eq_dfs_mean.actual_taxitime
    std_actual_taxitime = etd_eq_dfs_std.actual_taxitime
    fin_dict["average_actual_taxitime"] = average_actual_taxitime
    fin_dict["std_actual_taxitime"] = std_actual_taxitime

    # Iterate through unique columns and extract relevant statistics
    for curr_col in unique_cols:
        if curr_col not in etd_eq_dfs_mean:
            continue
        if "len" in curr_col:
            fin_dict[curr_col] = etd_eq_dfs_count[curr_col]
        else:
            fin_dict[curr_col] = etd_eq_dfs_mean[curr_col]
    return fin_dict


def extract_taxi_to_gate_time(curr_timepoint, df_arrival_standtimes):
    """
    Extracts the mean and standard deviation of taxi times to the gate for the given DataFrame
    and timepoint, using a variable lookback time based on the number of data points.

    Args:
        curr_timepoint (pd.Timestamp): The current timepoint to extract the statistics for.
        df_arrival_standtimes (pd.DataFrame): The DataFrame containing arrival standtimes data.

    Returns:
        np.array: A numpy array containing the count, mean, and standard deviation of taxi times
                  to the gate.
    """

    # Initialize lookback_time_mins and create a subset of the DataFrame
    lookback_time_mins = 60
    short_df_arrival_standtimes = df_arrival_standtimes[
        (df_arrival_standtimes.arrival_stand_actual_time < curr_timepoint)
        & (
            df_arrival_standtimes.arrival_stand_actual_time
            > curr_timepoint - np.timedelta64(lookback_time_mins, "m")
        )
    ]

    # Modify lookback_time_mins based on the number of data points
    if len(short_df_arrival_standtimes) == 0:
        lookback_time_mins = 180
    elif len(short_df_arrival_standtimes) < 3:
        lookback_time_mins = 120

    # Update the DataFrame subset if necessary
    if lookback_time_mins != 60:
        short_df_arrival_standtimes = df_arrival_standtimes[
            (df_arrival_standtimes.arrival_stand_actual_time < curr_timepoint)
            & (
                df_arrival_standtimes.arrival_stand_actual_time
                > curr_timepoint - np.timedelta64(lookback_time_mins, "m")
            )
        ]

    # Calculate mean and standard deviation of taxi times to the gate
    taxitime_to_gate_mean = np.mean(short_df_arrival_standtimes.taxitime_to_gate)
    taxitime_to_gate_std = np.std(short_df_arrival_standtimes.taxitime_to_gate)
    found_counts = len(short_df_arrival_standtimes)

    return np.array([found_counts, taxitime_to_gate_mean, taxitime_to_gate_std])


def split_gufi(curr_df):
    """
    Splits the 'gufi' column in the given DataFrame into separate columns for
    plane ID, departing airport code, arriving airport code, and airline code.

    Args:
        curr_df (pd.DataFrame): The DataFrame containing a 'gufi' column.

    Returns:
        pd.DataFrame: The updated DataFrame with the new columns.
    """

    # Split the 'gufi' column into separate columns for plane ID, departing airport code, and arriving airport code
    curr_df[
        ["plane_id", "departing_airport_code", "arriving_airport_code"]
    ] = curr_df.gufi.str.split(".", expand=True)[[0, 1, 2]]

    # Extract the airline code from the 'gufi' column
    curr_df["airline_code"] = curr_df.gufi.str[:3]

    return curr_df


def extract_etd_curr_airport(
    curr_load_dir, sav_dir, curr_airport, bool_submission_prep
):
    """
    Extracts estimated time of departure (ETD) information for a given airport and
    saves the results in a CSV file.

    Args:
        curr_load_dir (str): The directory containing the data files to be loaded.
        sav_dir (str): The directory where the resulting CSV file will be saved.
        curr_airport (str): The airport code for which the ETD information is to be extracted.
        bool_submission_prep (bool): A flag indicating if the function is used for submission preparation.

    Returns:
        None
    """

    df_train_labels, df_etd = data_loader_etd(curr_load_dir, curr_airport)

    if bool_submission_prep:
        df_train_labels = data_loader_submission_train_labels(
            curr_load_dir, curr_airport
        )

    ########unique timestamp preprocessing
    print(f"Feature Engineering for: {curr_airport} doing unique timestamp")
    df_final_labels_etd = df_train_labels.copy(deep=True)
    df_final_labels_etd = df_final_labels_etd.sort_values(
        by=["gufi", "timestamp"]
    ).reset_index(drop=True)
    df_etd_limited = (
        df_etd[df_etd.gufi.isin(df_final_labels_etd.gufi.unique())]
        .sort_values(by=["gufi", "timestamp"])
        .reset_index(drop=True)
    )
    df_final_labels_etd = (
        df_final_labels_etd[df_final_labels_etd.gufi.isin(df_etd.gufi.unique())]
        .sort_values(by=["gufi", "timestamp"])
        .reset_index(drop=True)
    )
    unique_planes = list(df_final_labels_etd.gufi.unique())
    df_etd_limited_no_dup = df_etd_limited.drop_duplicates(subset=["gufi"])
    new_planes_indexes_etd = list(df_etd_limited_no_dup.index)
    dropped_fin_labels = len(df_train_labels) - len(df_final_labels_etd)
    print(
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
    # df_final_labels_etd.head()

    curr_plane_counter = 1
    list_dicts_etd = []
    for index_counter, curr_train_labels_indx in enumerate(new_planes_indexes_labels):
        if index_counter % 25000 == 0 and index_counter == 0:
            print(
                f"---------{curr_airport}: {index_counter} out of {len(new_planes_indexes_etd)}---------"
            )
        elif index_counter % 25000 == 0:
            print(
                f"---------{curr_airport}: {np.around(index_counter/len(new_planes_indexes_etd) * 100,decimals=2)}%---------"
            )
        if len(new_planes_indexes_labels) - 1 == index_counter:
            break
        df_etd_limited_plane = df_etd_limited.iloc[
            new_planes_indexes_etd[curr_plane_counter - 1] : new_planes_indexes_etd[
                curr_plane_counter
            ]
        ]
        curr_gufi = df_final_labels_etd.gufi.iloc[curr_train_labels_indx]
        list_timepoints = list(
            df_final_labels_etd.timestamp.iloc[
                curr_train_labels_indx : new_planes_indexes_labels[index_counter + 1]
            ]
        )
        dep_times_dicts = extract_etdv3(
            curr_gufi, list_timepoints, df_etd_limited_plane
        )
        list_dicts_etd.extend(dep_times_dicts)
        curr_plane_counter += 1
    time_etd_df = time_etd_df.append(list_dicts_etd, ignore_index=True, sort=False)
    etd_sav_path = f"{sav_dir}timepointgufi_{curr_airport}_etd.csv"
    print(f"Saving the following file: {etd_sav_path}")
    time_etd_df[list_dicts_etd[0].keys()].to_csv(etd_sav_path, index=False)
