import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ensure import from correct path
sys.path.append(os.path.dirname(__file__))
from table_scripts import table_generation

sys.path.pop()


encoded_columns: tuple[str, ...] = (
    "airport",
    "departure_runways",
    "arrival_runways",
    "cloud",
    "lightning_prob",
    "precip",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
)


features: list[str] = [
    "gufi_flight_major_carrier",
    "deps_3hr",
    "deps_30hr",
    "arrs_3hr",
    "arrs_30hr",
    "deps_taxiing",
    "arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    "arr_taxi_30hr",
    "minute",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "airport",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
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


def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    with (solution_directory / "models.pickle").open("rb") as fp:
        model = pickle.load(fp)
    with (solution_directory / "encoders.pickle").open("rb") as fp:
        encoders = pickle.load(fp)
    return model, encoders


def predict(
    config: pd.DataFrame,
    etd: pd.DataFrame,
    first_position: pd.DataFrame,
    lamp: pd.DataFrame,
    mfs: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tbfm: pd.DataFrame,
    tfm: pd.DataFrame,
    airport: str,
    prediction_time: pd.Timestamp,
    partial_submission_format: pd.DataFrame,
    model: Any,
    solution_directory: Path,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time."""

    if len(partial_submission_format) == 0:
        return partial_submission_format

    model, encoders = model[0], model[1]

    _df: pd.DataFrame = partial_submission_format.copy()

    feature_tables: dict[str, pd.DataFrame] = {
        "etd": etd.sort_values("timestamp"),
        "config": config.sort_values("timestamp", ascending=False),
        "first_position": first_position,
        "lamp": lamp.set_index("timestamp", drop=False).sort_index(),
        "runways": runways,
        "standtimes": standtimes,
        "mfs": mfs,
    }

    _df = table_generation.add_all_features(_df, feature_tables, True)

    _df["precip"] = _df["precip"].astype(str)

    for col in encoded_columns:
        _df[[col]] = encoders[col].transform(_df[[col]].values)

    prediction = partial_submission_format.copy()

    prediction["minutes_until_pushback"] = model[airport].predict(
        _df[features], categorical_features="auto"
    )

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(
        lower=0
    ).fillna(0)

    return prediction
