import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
from config import *
from run_inference import load_model, predict


# Function was provided by the competition organizers for inference
def _load_features(airport: str, feature_name: str) -> pd.DataFrame:
    if feature_name in FEATURES[airport]:
        return FEATURES[airport][feature_name]
    else:
        FEATURES[airport][feature_name] = pd.read_csv(
            data_directory / airport / f"{airport}_{feature_name}.csv.bz2",
            **read_csv_kwargs[feature_name],
        )
        return FEATURES[airport][feature_name]


# Function was provided by the competition organizers for inference
def _get_censored_features(
    airport: str, prediction_time: datetime
) -> Dict[str, pd.DataFrame]:
    """Return the features for an airport. Handles reading the full feature tables from disk and
    using in-memory versions for subsequent calls."""

    gufis = set()
    censored_features = {}
    for feature_name in [
        "config",
        "etd",
        "first_position",
        "lamp",
        "mfs",
        "runways",
        "standtimes",
        "tbfm",
        "tfm",
    ]:
        if feature_name == "mfs":
            continue

        features = _load_features(airport, feature_name)
        censored_features[feature_name] = features.loc[
            (features.timestamp > (prediction_time - timedelta(hours=30)))
            & (features.timestamp <= prediction_time)
        ]

        if "gufi" in censored_features[feature_name]:
            gufis.update(set(censored_features[feature_name].gufi.tolist()))

    mfs = _load_features(airport, "mfs")
    censored_features["mfs"] = mfs.loc[mfs.gufi.isin(gufis)]

    return (
        censored_features["config"],
        censored_features["etd"],
        censored_features["first_position"],
        censored_features["lamp"],
        censored_features["mfs"],
        censored_features["runways"],
        censored_features["standtimes"],
        censored_features["tbfm"],
        censored_features["tfm"],
    )


# Function was provided by the competition organizers for inference
def _check_partial_predictions(partial_submission_format, partial_predictions):
    if not partial_submission_format.drop(columns=["minutes_until_pushback"]).equals(
        partial_predictions.drop(columns=["minutes_until_pushback"])
    ):
        error_message = "Partial submission does not match expected format."

        missing_columns = partial_submission_format.columns.difference(
            partial_predictions.columns
        )
        if len(missing_columns):
            error_message += (
                f""" {len(missing_columns)} missing column(s) """
                f"""({", ".join(str(k) for k in missing_columns[:3])}...)."""
            )

        extra_columns = partial_predictions.columns.difference(
            partial_submission_format.columns
        )
        if len(extra_columns):
            error_message += (
                f""" {len(extra_columns)} extra column(s) """
                f"""({", ".join(str(k) for k in extra_columns[:3])}...)."""
            )

        missing_keys = partial_submission_format.index.difference(
            partial_predictions.index
        )
        if len(missing_keys):
            error_message += (
                f""" {len(missing_keys):,} missing indices """
                f"""({", ".join(str(k) for k in missing_keys[:3])}...)."""
            )

        extra_keys = partial_predictions.index.difference(
            partial_submission_format.index
        )
        if len(extra_keys):
            error_message += (
                f""" {len(extra_keys):,} extra indices """
                f"""({", ".join(str(k) for k in extra_keys[:3])}...)."""
            )
        raise KeyError(error_message)


# Function was provided by the competition organizers for inference
def _log_progress(iterable, message: str, every: int):
    for i, item in enumerate(iterable):
        if (i % every) == 0:
            print(message.format(i=i + 1, n=len(iterable), pct=i / len(iterable)))
        yield item


# Function was provided by the competition organizers for inference
def compute_predictions():
    """Load any model assets from disk."""
    model = load_model(model_directory)

    submission_format = pd.read_csv(submission_format_path, parse_dates=["timestamp"])
    prediction_times = pd.to_datetime(
        submission_format.timestamp.unique()
    ).sort_values()

    for prediction_time in _log_progress(
        prediction_times,
        message="Predicting {i} of {n} ({pct:.2%}) timestamps",
        every=1,
    ):
        for airport in airports:
            (
                config,
                etd,
                first_position,
                lamp,
                mfs,
                runways,
                standtimes,
                tbfm,
                tfm,
            ) = _get_censored_features(airport, prediction_time)

            # Partial submission format
            partial_submission_format = submission_format.loc[
                (submission_format.timestamp == prediction_time)
                & (submission_format.airport == airport)
            ].reset_index(drop=True)

            # Call participant's predict function
            partial_predictions = predict(
                config,
                etd,
                first_position,
                lamp,
                mfs,
                runways,
                standtimes,
                tbfm,
                tfm,
                airport,
                prediction_time,
                partial_submission_format,
                model,
            )

            # Check that partial predictions match partial submission format
            _check_partial_predictions(partial_submission_format, partial_predictions)

            partial_prediction_path = (
                prediction_directory / f"{airport} {prediction_time}.csv"
            )

            # Save partial predictions
            partial_predictions.to_csv(
                partial_prediction_path, date_format="%Y-%m-%d %H:%M:%S", index=False
            )


# Function was provided by the competition organizers for inference
def postprocess_predictions():
    print("Concatenating partial predictions")
    submission = pd.concat(
        pd.read_csv(path, index_col=["gufi", "timestamp", "airport"])
        for path in prediction_directory.glob("*.csv")
    )

    print("Reindexing submission to match submission format")
    submission_format = pd.read_csv(
        submission_format_path, index_col=["gufi", "timestamp", "airport"]
    )

    submission = submission.loc[submission_format.index]

    print("Casting prediction to integer")
    submission["minutes_until_pushback"] = submission.minutes_until_pushback.astype(int)

    print("Saving submission")
    submission.to_csv(submission_path, date_format="%Y-%m-%d %H:%M:%S")
