from pathlib import Path
from typing import Any
from loguru import logger

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from .utilities import catboost_predictions, baseline_predictions


def load_model(solution_directory: Path) -> Any:
    airports = [
        "katl",
        "kclt",
        "kden",
        "kdfw",
        "kjfk",
        "kmem",
        "kmia",
        "kord",
        "kphx",
        "ksea",
    ]

    models = {}
    for airport in airports:
        models[airport.upper()] = {}
        for version in [0, 1, 2]:
            current_model = CatBoostRegressor()
            current_model.load_model(
                solution_directory / f"models/{airport}_model_v{version}"
            )
            models[airport.upper()][version] = current_model

        model = CatBoostRegressor()
        model.load_model(solution_directory / f"models/global_model")
        models[airport.upper()]["global_model"] = model

    return models


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

    try:
        predictions = catboost_predictions(
            prediction_time,
            partial_submission_format,
            mfs,
            config,
            etd,
            lamp,
            runways,
            standtimes,
            model[airport],
            airport,
        )
    except:
        try:
            logger.info(f"Defaulting to baseline predictions")
            predictions = baseline_predictions(partial_submission_format, etd)
        except:
            logger.info(f"Defaulting to manual predictions")
            predictions = partial_submission_format
            predictions["minutes_until_pushback"] = 30

    return predictions
