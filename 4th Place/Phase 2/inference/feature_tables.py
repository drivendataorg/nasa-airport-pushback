#
# Author: Yudong Lin
#
# read all feature tables

import os
from collections import namedtuple

import pandas as pd
from utils import get_csv_path

AIRLINES: tuple[str, ...] = (
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
)


def get_public_feature_tables(data_dir: str, _airport: str) -> dict[str, pd.DataFrame]:
    return {
        "etd": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_etd.csv"),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp"),
        "config": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_config.csv"),
            parse_dates=["timestamp", "start_time"],
            dtype={"departure_runways": "category", "arrival_runways": "category"},
        ).sort_values("timestamp", ascending=False),
        "first_position": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]
        ),
        "lamp": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_lamp.csv"),
            parse_dates=["timestamp", "forecast_timestamp"],
            dtype={
                "cloud_ceiling": "int8",
                "visibility": "int8",
                "cloud": "category",
                "lightning_prob": "category",
                "precip": "category",
            },
        )
        .set_index("timestamp", drop=False)
        .sort_index(),
        "runways": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_runways.csv"),
            dtype={"departure_runway_actual": "category", "arrival_runway_actual": "category"},
            parse_dates=["timestamp", "departure_runway_actual_time", "arrival_runway_actual_time"],
        ),
        "mfs": pd.read_csv(get_csv_path(data_dir, "public", _airport, f"{_airport}_mfs.csv")),
        "standtimes": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_standtimes.csv"),
            parse_dates=["departure_stand_actual_time", "timestamp"],
        ).sort_values("timestamp"),
    }


PrivateAirlineFeature = namedtuple("PrivateAirlineFeature", ["mfs", "standtimes"])


def get_private_feature_tables(data_dir: str, _airport: str) -> dict[str, PrivateAirlineFeature]:
    return {_airline: try_load_csv_with_private_feature(data_dir, _airport, _airline) for _airline in AIRLINES}


def try_load_csv_with_private_feature(data_dir: str, _airport: str, airline: str) -> PrivateAirlineFeature:
    _path: str = get_csv_path(data_dir, "private", _airport, f"{_airport}_{airline}_mfs.csv")
    mfs_pd: pd.DataFrame = (
        pd.read_csv(
            _path,
            dtype={
                "aircraft_engine_class": "category",
                "aircraft_type": "category",
                "major_carrier": "category",
                "flight_type": "category",
                "isdeparture": bool,
            },
        )
        if os.path.exists(_path)
        else pd.DataFrame(
            columns=["gufi", "aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "isdeparture"],
            dtype={
                "aircraft_engine_class": "category",
                "aircraft_type": "category",
                "major_carrier": "category",
                "flight_type": "category",
                "isdeparture": bool,
            },
        )
    )
    _path = get_csv_path(data_dir, "private", _airport, f"{_airport}_{airline}_standtimes.csv")
    standtimes_pd: pd.DataFrame = (
        pd.read_csv(
            _path,
            parse_dates=["timestamp", "arrival_stand_actual_time", "departure_stand_actual_time"],
        )
        if os.path.exists(_path)
        else pd.DataFrame(columns=["gufi", "timestamp", "arrival_stand_actual_time", "departure_stand_actual_time"])
    )
    return PrivateAirlineFeature(mfs_pd, standtimes_pd)
