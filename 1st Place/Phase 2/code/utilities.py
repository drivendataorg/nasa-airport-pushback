import re
import time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from config import (
    AIRCRAFT_TYPES,
    AIRPORTS,
    CARRIER_TYPES,
    END_HOLIDAY,
    ENGINE_CLASSES,
    FLIGHT_IDS,
    FLIGHT_NUMBERS,
    FLIGHT_TYPES,
    OBJECT_COLS,
    RUNWAYS,
    START_HOLIDAY,
)


def extract_mfs_features(pairs: pd.DataFrame, mfs: pd.DataFrame) -> pd.DataFrame:
    print(time.ctime(), "Extracting MFS features")
    # Define base of pairs of gufi-timestamp for which we want the features
    base = pairs[["gufi", "timestamp"]].copy()

    # Add variables intrinsic to the GUFI
    base["mfs_cat_flightid"] = base["gufi"].apply(lambda x: str(x).split(".")[0])
    base.loc[
        base["mfs_cat_flightid"].isin(FLIGHT_IDS) == False, "mfs_cat_flightid"
    ] = "OTHER"
    base["mfs_cat_airline"] = (
        base["gufi"]
        .apply(lambda x: re.sub(r"[0-9]+", "", x.split(".")[0]))
        .fillna("OTHER")
    )
    base["mfs_cat_flightno"] = base["mfs_cat_flightid"].apply(
        lambda x: "".join(filter(str.isdigit, str(x)))
    )
    base.loc[
        base["mfs_cat_flightno"].isin(FLIGHT_NUMBERS) == False, "mfs_cat_flightno"
    ] = "OTHER"
    base["mfs_flightno_length"] = base["mfs_cat_flightno"].apply(lambda x: len(str(x)))
    base["mfs_cat_no_last"] = base["mfs_cat_flightno"].apply(
        lambda x: "OTHER" if len(str(x)) == 0 else str(x)[-1]
    )
    base["mfs_cat_arrival"] = base["gufi"].apply(lambda x: x.split(".")[2])
    base["mfs_cat_last"] = base["gufi"].apply(lambda x: x.split(".")[-1])
    base["mfs_cat_sectolast"] = (
        base["gufi"].apply(lambda x: x.split(".")[-2]).astype(str)
    )
    base.loc[
        base["mfs_cat_arrival"].isin(AIRPORTS) == False, "mfs_cat_arrival"
    ] = "OTHER"
    base["mfs_dep_time"] = base["gufi"].map(
        dict(
            zip(
                mfs["gufi"],
                pd.to_datetime(
                    mfs["gufi"].apply(lambda x: x.split(".")[3] + x.split(".")[4]),
                    format="%y%m%d%H%M",
                ),
            )
        )
    )
    base["mfs_cat_hourdep"] = base["mfs_dep_time"].dt.hour
    base.loc[base["mfs_dep_time"] > base["timestamp"], "mfs_time_diff"] = (
        -(base["mfs_dep_time"] - base["timestamp"]).dt.total_seconds() / 60
    )
    base.loc[base["mfs_dep_time"] <= base["timestamp"], "mfs_time_diff"] = (
        base["timestamp"] - base["mfs_dep_time"]
    ).dt.total_seconds() / 60
    base.drop(columns="mfs_dep_time", inplace=True)

    # Add aircraft engine class feature
    base["mfs_cat_engineclass"] = base["gufi"].map(
        dict(zip(mfs["gufi"], mfs["aircraft_engine_class"]))
    )
    base.loc[
        base["mfs_cat_engineclass"].isin(ENGINE_CLASSES) == False, "mfs_cat_engineclass"
    ] = "OTHER"

    # Add aircraft type feature
    base["mfs_cat_type"] = base["gufi"].map(
        dict(zip(mfs["gufi"], mfs["aircraft_type"]))
    )
    base.loc[
        base["mfs_cat_type"].isin(AIRCRAFT_TYPES) == False, "mfs_cat_type"
    ] = "OTHER"

    # Add major carrier feature
    base["mfs_cat_majorcarrier"] = base["gufi"].map(
        dict(zip(mfs["gufi"], mfs["major_carrier"]))
    )
    base.loc[
        base["mfs_cat_majorcarrier"].isin(CARRIER_TYPES) == False,
        "mfs_cat_majorcarrier",
    ] = "OTHER"

    # Add flight type feature
    base["mfs_cat_flighttype"] = base["gufi"].map(
        dict(zip(mfs["gufi"], mfs["flight_type"]))
    )
    base.loc[
        base["mfs_cat_flighttype"].isin(FLIGHT_TYPES) == False, "mfs_cat_flighttype"
    ] = "OTHER"

    # Add departure flag
    base["mfs_departure"] = base["gufi"].map(dict(zip(mfs["gufi"], mfs["isdeparture"])))

    return base


def extract_config_features(
    config: pd.DataFrame, start_time: str, end_time: str
) -> pd.DataFrame:
    print(time.ctime(), "Extracting Config features")
    # Sort configuration of airport based on timestamp
    config = config.sort_values("timestamp")

    # Make start and end time timestamps
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # Keep only records between start and end times
    config = config[
        (config["timestamp"] >= start_time) & (config["timestamp"] <= end_time)
    ]

    # Expand configuration of airport to be from start to end dates
    config = (
        pd.concat(
            [
                pd.DataFrame({"timestamp": [start_time]}),
                config,
                pd.DataFrame({"timestamp": [end_time]}),
            ]
        )
        .bfill()
        .ffill()
    )
    config = config.groupby("timestamp").first().reset_index()

    # Resample configuration of airport to be at a 5minute level
    config = (
        config.set_index("timestamp")[["departure_runways", "arrival_runways"]]
        .resample("5min")
        .ffill()
        .bfill()
        .reset_index()
    )

    # Extract features about the current runways
    def process(x, k):
        if k not in x:
            return 0
        if f"{k}R" in x:
            return 1
        if f"{k}L" in x:
            return 2
        if f"{k}C" in x:
            return 3
        return 4

    for r in RUNWAYS:
        config[f"config_arrival_cat_{r}"] = config["arrival_runways"].apply(
            lambda x: process(str(x), r)
        )
        config[f"config_departure_cat_{r}"] = config["departure_runways"].apply(
            lambda x: process(str(x), r)
        )

    # Extract features about number of active runways
    config["config_n_active_dep"] = (
        config["departure_runways"].apply(lambda x: str(x).count(",")) + 1
    )
    config["config_n_active_arr"] = (
        config["arrival_runways"].apply(lambda x: str(x).count(",")) + 1
    )

    # Extract features about changes in configuration
    config["config_change_dep"] = config["departure_runways"].ne(
        config["departure_runways"].shift().bfill()
    )
    config["config_change_arr"] = config["arrival_runways"].ne(
        config["arrival_runways"].shift().bfill()
    )

    # Extract features about past changes in look-back windows
    for p in [6, 12, 24, 36, 48, 120]:
        config[f"config_n_chang_dep_last_{p}"] = (
            config["config_change_dep"].rolling(p).sum().fillna(0)
        )
        config[f"config_n_chang_arr_last_{p}"] = (
            config["config_change_arr"].rolling(p).sum().fillna(0)
        )

    # Drop non-feature columns
    config.drop(columns=["departure_runways", "arrival_runways"], inplace=True)

    return config


def extract_etd_features(pairs: pd.DataFrame, etd: pd.DataFrame) -> pd.DataFrame:
    print(time.ctime(), "Extracting ETD features")
    # Define base of pairs of gufi-timestamp for which we want the features
    base = pairs[["gufi", "timestamp"]].copy()
    base = base.sort_values("timestamp")

    # Filter etd table
    etd = etd[etd["gufi"].isin(base["gufi"].unique())]

    # Resample etd at a 15-minute level and keep the last record
    etd = etd.sort_values("timestamp")
    etd["timestamp"] = etd["timestamp"].dt.ceil("15min")
    etd = etd.groupby(["gufi", "timestamp"]).last().reset_index()

    # Combine base and etd tables
    etd = pd.concat([etd, base])

    # Among duplicate gufi-timestamp pairs, keep that with informed etd values
    etd = etd.groupby(["gufi", "timestamp"]).first().reset_index()

    # Forward fill missing values
    etd["departure_runway_estimated_time"] = etd.groupby("gufi")[
        "departure_runway_estimated_time"
    ].ffill()

    # Add values about changes in estimated time at runway
    etd = etd.sort_values(["gufi", "timestamp", "departure_runway_estimated_time"])

    etd.loc[
        etd["departure_runway_estimated_time"].shift()
        > etd["departure_runway_estimated_time"],
        "etd_est_change",
    ] = (
        -(
            etd["departure_runway_estimated_time"].shift()
            - etd["departure_runway_estimated_time"]
        ).dt.total_seconds()
        / 60
    )
    etd.loc[
        etd["departure_runway_estimated_time"].shift()
        <= etd["departure_runway_estimated_time"],
        "etd_est_change",
    ] = (
        (
            etd["departure_runway_estimated_time"]
            - etd["departure_runway_estimated_time"].shift()
        ).dt.total_seconds()
        / 60
    )

    etd.loc[etd["gufi"] != etd["gufi"].shift(), "etd_est_change"] = 0

    # Add delay with respect to first runway estimated time observed
    etd["etd_first_etd"] = etd["gufi"].map(
        etd.groupby("gufi")["departure_runway_estimated_time"].first()
    )
    etd.loc[
        etd["departure_runway_estimated_time"] > etd["etd_first_etd"],
        "etd_diff_withfirst",
    ] = (
        (
            etd["departure_runway_estimated_time"] - etd["etd_first_etd"]
        ).dt.total_seconds()
        / 60
    )
    etd.loc[
        etd["departure_runway_estimated_time"] <= etd["etd_first_etd"],
        "etd_diff_withfirst",
    ] = (
        -(
            etd["etd_first_etd"] - etd["departure_runway_estimated_time"]
        ).dt.total_seconds()
        / 60
    )
    etd = etd.drop(columns="etd_first_etd")

    # Calculate time till estimated departure
    etd.loc[
        etd["departure_runway_estimated_time"] > etd["timestamp"],
        "etd_time_till_est_dep",
    ] = (
        (etd["departure_runway_estimated_time"] - etd["timestamp"]).dt.total_seconds()
        / 60
    )
    etd.loc[
        etd["departure_runway_estimated_time"] <= etd["timestamp"],
        "etd_time_till_est_dep",
    ] = (
        -(etd["timestamp"] - etd["departure_runway_estimated_time"]).dt.total_seconds()
        / 60
    )

    res = etd.set_index(["gufi", "timestamp"]).copy()
    etd = etd.set_index("timestamp")

    periods = [60, 120, 240, 360, 720, 1440]
    for p in periods:
        s = etd.groupby("gufi").rolling(f"{p}min").etd_est_change
        aggs = s.agg({"sum", "max", "std"}).rename(
            columns={
                "sum": f"etd_roll_{p}_sum",
                "max": f"etd_roll_{p}_max",
                "std": f"etd_roll_{p}_std",
            }
        )
        for c in aggs.columns:
            res[c] = aggs[c]

    res = res.reset_index()

    # Keep only relevant records to our base of pairs of gufi-timestamp
    res = res.drop(columns="departure_runway_estimated_time")

    return res


def extract_moment_features(pairs: pd.DataFrame) -> pd.DataFrame:
    print(time.ctime(), "Extracting Moment features")
    base = pairs[["timestamp"]].copy()
    base = base.drop_duplicates()

    # Moment of day features
    base["mom_cat_hourofday"] = base["timestamp"].dt.hour
    base["mom_cat_dayofweek"] = base["timestamp"].dt.dayofweek
    base["mom_cat_month"] = base["timestamp"].dt.month
    base["mom_dayofmonth"] = base["timestamp"].dt.day
    base["mom_cat_woy"] = base["timestamp"].dt.isocalendar().week
    base["mom_cat_wom"] = np.ceil(base["mom_dayofmonth"] / 7).astype(int)

    # Holiday features
    holidays = [
        str(c)[:10] for c in calendar().holidays(start=START_HOLIDAY, end=END_HOLIDAY)
    ]
    base["mom_cat_isholiday"] = base["timestamp"].apply(
        lambda x: str(x)[:10] in holidays
    )
    holidays = pd.to_datetime(holidays)
    base["mom_days_until_holiday"] = (
        base["timestamp"]
        .apply(lambda x: min([y for y in holidays if y >= x]) - x)
        .dt.days
    )

    return base


def extract_weather_features(
    lamp: pd.DataFrame, start_time: str, end_time: str
) -> pd.DataFrame:
    print(time.ctime(), "Extracting Weather features")
    # Sort configuration of airport based on timestamp
    lamp = lamp.sort_values(["timestamp", "forecast_timestamp"])

    # Make start and end time timestamps
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # Keep only records between start and end times
    lamp = lamp[(lamp["timestamp"] >= start_time) & (lamp["timestamp"] <= end_time)]

    # Expand configuration of airport to be from start to end dates
    lamp = (
        pd.concat(
            [
                pd.DataFrame({"timestamp": [start_time]}),
                lamp,
                pd.DataFrame({"timestamp": [end_time]}),
            ]
        )
        .ffill()
        .bfill()
    )

    # Adjust contents of raw lamp table
    lamp["lightning_prob"] = lamp["lightning_prob"].map(
        {"L": 0, "M": 1, "N": 2, "H": 3}
    )
    lamp["cloud"] = (
        lamp["cloud"].map({"OV": 4, "BK": 3, "CL": 0, "FW": 1, "SC": 2}).fillna(3)
    )
    lamp["precip"] = lamp["precip"].astype(float)
    lamp["time_ahead_prediction"] = (
        lamp["forecast_timestamp"] - lamp["timestamp"]
    ).dt.total_seconds() / 3600
    lamp.sort_values(["timestamp", "time_ahead_prediction"], inplace=True)

    # Create past weather features
    weather = (
        lamp.groupby("timestamp")
        .first()
        .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
    )
    weather = weather.rolling("6h").agg({"mean", "min", "max"}).reset_index()
    weather.columns = [
        "temp_" + c[0] + "_" + c[1] + "_last6h" if c[0] != "timestamp" else "timestamp"
        for c in weather.columns
    ]

    # Resample at a 15minute interval
    weather = weather.set_index("timestamp").resample("15min").ffill().reset_index()

    # Merge data for the upcoming 3h worth of predictions
    next_3h = (
        lamp[(lamp["time_ahead_prediction"] <= 3.5)]
        .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
        .groupby("timestamp")
        .mean()
        .reset_index()
    )
    next_3h.columns = [
        "temp_" + c + "_next3" if c != "timestamp" else "timestamp"
        for c in next_3h.columns
    ]
    next_3h = next_3h.set_index("timestamp").resample("15min").ffill().reset_index()
    weather = weather.merge(next_3h, how="left", on="timestamp")

    # Merge data for the upcoming 3-6h worth of predictions
    next_6h = (
        lamp[
            (lamp["time_ahead_prediction"] > 3.5)
            & (lamp["time_ahead_prediction"] <= 6.5)
        ]
        .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
        .groupby("timestamp")
        .mean()
        .reset_index()
    )
    next_6h.columns = [
        "temp_" + c + "_next36" if c != "timestamp" else "timestamp"
        for c in next_6h.columns
    ]
    next_6h = next_6h.set_index("timestamp").resample("15min").ffill().reset_index()
    weather = weather.merge(next_6h, how="left", on="timestamp")

    # Create change features from the past
    temp_feats = [
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ]
    for i in range(1, 7):
        for f in temp_feats:
            lamp[f"change_{f}_{i}"] = lamp[f] - lamp.groupby("forecast_timestamp")[
                f
            ].shift(i)

    past_change_feats = (
        lamp.groupby("forecast_timestamp")
        .last()
        .drop(columns=["timestamp", "time_ahead_prediction"])
    )
    past_change_feats.columns = ["temp_" + c for c in past_change_feats.columns]
    past_change_feats = past_change_feats.reset_index().rename(
        columns={"forecast_timestamp": "timestamp"}
    )

    weather = (
        weather.merge(past_change_feats, how="left", on="timestamp").ffill().bfill()
    )

    # Create change features from the future
    future_change_feats = weather[["timestamp"]].copy()
    temp_feats = [
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ]
    for i in [1, 3, 6, 9, 12, 24]:
        current = lamp[lamp["time_ahead_prediction"] == 0.5].copy()
        future = lamp[lamp["time_ahead_prediction"] == i + 0.5].copy()
        for f in temp_feats:
            future_change_feats[f"temp_futchange_{f}_{i}"] = future_change_feats[
                "timestamp"
            ].map(dict(zip(current["timestamp"], current[f]))) - future_change_feats[
                "timestamp"
            ].map(
                dict(zip(future["timestamp"], future[f]))
            )

    future_change_feats = future_change_feats.ffill().bfill()

    weather = weather.merge(future_change_feats, how="left", on="timestamp")

    return weather


def extract_runway_features(
    runways: pd.DataFrame, start_time: str, end_time: str
) -> pd.DataFrame:
    print(time.ctime(), "Extracting Runway features")
    # Sort runways data by timestamp
    runways = runways.sort_values("timestamp")

    # Create grid of dates to create the features of
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    dates = pd.DataFrame(
        {"timestamp": pd.date_range(start_time, end_time, freq="15min")}
    )
    dates["selected"] = 1

    # Split runways in departure and arrivals
    dep_runways = runways[runways["departure_runway_actual"].isna() == False].copy()
    dep_runways = pd.concat([dep_runways, dates]).sort_values("timestamp")
    dep_runways = dep_runways.set_index("timestamp")

    arr_runways = runways[runways["arrival_runway_actual"].isna() == False].copy()
    arr_runways = pd.concat([arr_runways, dates]).sort_values("timestamp")
    arr_runways = arr_runways.set_index("timestamp")

    # Extract counts of arrivals and departures at different lookback periods
    for p in [15, 30, 60, 90, 120, 360, 720, 1440]:
        dep_runways[f"runways_{p}_totdep"] = (
            dep_runways.rolling(f"{p}min").gufi.count().astype(int)
        )
        arr_runways[f"runways_{p}_totarr"] = (
            arr_runways.rolling(f"{p}min").gufi.count().astype(int)
        )

    # Stick only with created features
    dep_runways = (
        dep_runways[dep_runways["selected"] == 1]
        .drop(
            columns=[
                "gufi",
                "departure_runway_actual",
                "departure_runway_actual_time",
                "arrival_runway_actual",
                "arrival_runway_actual_time",
                "selected",
            ]
        )
        .reset_index()
    )
    arr_runways = (
        arr_runways[arr_runways["selected"] == 1]
        .drop(
            columns=[
                "gufi",
                "departure_runway_actual",
                "departure_runway_actual_time",
                "arrival_runway_actual",
                "arrival_runway_actual_time",
                "selected",
            ]
        )
        .reset_index()
    )

    # Combine departure and arrival runway data
    runway_feats = dep_runways.merge(arr_runways, how="left", on="timestamp")

    return runway_feats


def extract_standtime_features(pairs, standtimes, etd):
    print(time.ctime(), "Extracting Standtime features")
    # Select only the timestamps in our perimeter
    base = pairs[["timestamp"]].drop_duplicates()

    # Sort by timestamps and clip at a 15min level
    standtimes = standtimes.sort_values("timestamp")
    standtimes["timestamp"] = standtimes["timestamp"].dt.ceil("15min")

    # Add first departure estimation and delay variables
    standtimes["first_dep_estimation"] = standtimes["gufi"].map(
        etd.sort_values("timestamp")
        .groupby("gufi")
        .departure_runway_estimated_time.first()
    )
    standtimes.loc[
        standtimes["departure_stand_actual_time"] >= standtimes["first_dep_estimation"],
        "delay",
    ] = (
        (
            standtimes["departure_stand_actual_time"]
            - standtimes["first_dep_estimation"]
        ).dt.total_seconds()
        / 60
    )
    standtimes.loc[
        standtimes["departure_stand_actual_time"] < standtimes["first_dep_estimation"],
        "delay",
    ] = (
        -(
            standtimes["first_dep_estimation"]
            - standtimes["departure_stand_actual_time"]
        ).dt.total_seconds()
        / 60
    )

    # Append timestamps to standtimes table
    standtimes = pd.concat([standtimes, base])
    standtimes = standtimes.sort_values("timestamp")

    # Extract features at different lookback periods
    periods = [1, 2, 4, 6, 12, 24]
    for p in periods:
        r = (
            standtimes.set_index("timestamp")
            .rolling(f"{p}h")
            .agg({"delay": ["max", "mean"], "gufi": "count"})
            .reset_index(level=0)
        )
        r.columns = [
            "timestamp",
            f"standtimes_maxdelay_{p}",
            f"standtimes_meandelay_{p}",
            f"standtimes_count_{p}",
        ]
        r = r.groupby("timestamp").last().reset_index()
        base = base.merge(r, how="left", on="timestamp")

    return base


def adjust_master(master: pd.DataFrame) -> pd.DataFrame:
    print(time.ctime(), "Adjusting master table")
    # NaN imputation
    master = master.ffill().bfill().fillna(0)

    # Variable type change to make the master table more lightweight
    for c in master.columns:
        if c not in (
            OBJECT_COLS + ["gufi", "timestamp", "airport", "minutes_until_pushback"]
        ):
            master[c] = master[c].astype("int16")

    return master


def generate_predictions(model, master, airport) -> pd.DataFrame:
    for v in [0, 1, 2]:
        # Retrieve model predictions
        master[f"pred_{v}"] = model[v].predict(master[model[v].feature_names_])
        # Adjust predictions
        if v in [0, 2]:
            master[f"pred_{v}"] = (
                (master[f"pred_{v}"] + master["etd_time_till_est_dep"])
                .clip(lower=1, upper=299)
                .astype(int)
            )
        else:
            master[f"pred_{v}"] = (
                master[f"pred_{v}"].clip(lower=1, upper=299).astype(int)
            )

    master["feat_cat_airport"] = airport.lower()
    master["global_pred"] = model["global_model"].predict(
        master[model["global_model"].feature_names_]
    )

    master["minutes_until_pushback"] = (
        (
            (
                master[f"pred_0"]
                + master[f"pred_1"]
                + master[f"pred_2"]
                + master[f"global_pred"]
            )
            / 4
        )
        .clip(lower=4, upper=260)
        .astype(int)
    )

    if airport.upper() in ["KPHX", "KMIA", "KJFK"]:
        master["minutes_until_pushback"] = (
            master[f"pred_1"].clip(lower=4, upper=260).astype(int)
        )

    master = master[["gufi", "timestamp", "airport", "minutes_until_pushback"]]

    return master


def catboost_predictions(
    prediction_time,
    partial_submission_format,
    mfs,
    config,
    etd,
    lamp,
    runways,
    standtimes,
    model,
    airport,
):
    # Test the case where partial submission format is empty
    if partial_submission_format.shape[0] == 0:
        return partial_submission_format

    # Define a start and end times
    start_time = str(prediction_time - pd.Timedelta("26h"))
    end_time = str(prediction_time)

    # Define the perimeter for which we want to extract features
    perimeter = partial_submission_format[["gufi", "timestamp"]].copy()

    # Extract all the features
    mfs_features = extract_mfs_features(perimeter, mfs)
    config_features = extract_config_features(config, start_time, end_time)
    etd_features = extract_etd_features(perimeter, etd)
    mom_features = extract_moment_features(perimeter)
    lamp_features = extract_weather_features(lamp, start_time, end_time)
    runway_features = extract_runway_features(runways, start_time, end_time)
    standtime_features = extract_standtime_features(perimeter, standtimes, etd)

    # Combine the features in a single master table
    master = partial_submission_format[["gufi", "timestamp", "airport"]].copy()
    master = master.merge(mfs_features, how="left", on=["gufi", "timestamp"])
    master = master.merge(config_features, how="left", on="timestamp")
    master = master.merge(etd_features, how="left", on=["gufi", "timestamp"])
    master = master.merge(mom_features, how="left", on="timestamp")
    master = master.merge(lamp_features, how="left", on="timestamp")
    master = master.merge(runway_features, how="left", on="timestamp")
    master = master.merge(standtime_features, how="left", on="timestamp")

    # Fill NaN values and adjust variable types
    master = adjust_master(master)

    # Retrieve predictions
    predictions = generate_predictions(model, master, airport)

    return predictions


def baseline_predictions(partial_submission_format, etd) -> pd.DataFrame:
    latest_etd = (
        etd.sort_values("timestamp")
        .groupby("gufi")
        .last()
        .departure_runway_estimated_time
    )
    departure_runway_estimated_time = partial_submission_format.merge(
        latest_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    prediction = partial_submission_format.copy()
    prediction["minutes_until_pushback"] = (
        (
            departure_runway_estimated_time - partial_submission_format.timestamp
        ).dt.total_seconds()
        / 60
    ) - 30

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(
        lower=1, upper=299
    ).fillna(15)

    return prediction
