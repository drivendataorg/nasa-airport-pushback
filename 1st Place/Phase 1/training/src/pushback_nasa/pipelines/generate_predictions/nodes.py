"""
This is a boilerplate pipeline 'generate_predictions'
generated using Kedro 0.18.4
"""

import pandas as pd


def retrieve_predictions(
    sample_sub: pd.DataFrame, master: pd.DataFrame, model, airport: str
) -> pd.DataFrame:
    # Select slice of sample_sub containing the current airport
    current = sample_sub[sample_sub["airport"] == airport].copy()

    # Combine pairs of gufi-timestamp from current with the features of master
    current = current.merge(master, how="left", on=["gufi", "timestamp"])

    # Generate predictions
    current["minutes_until_pushback"] = (
        model.predict(current[model.feature_names_]) + current["etd_time_till_est_dep"]
    )

    # Change type of predicted column
    current["minutes_until_pushback"] = round(current["minutes_until_pushback"]).astype(
        int
    )

    # Select columns to submit
    current = current[["gufi", "timestamp", "airport", "minutes_until_pushback"]]

    return current


def combine_predictions(*args) -> pd.DataFrame:
    submission = pd.concat(list(args))

    return submission
