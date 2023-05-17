import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from catboost import Pool, CatBoostRegressor

from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from make_dataset import make_dataset, get_negative_predictions
from config import *


def train_new_model(data_df):
    print(f"Training model for {airport}")
    X = data_df.drop(["minutes_until_pushback"], axis=1)
    y = data_df["minutes_until_pushback"]

    cat_cols = [
        "aircraft_engine_class",
        "major_carrier",
        "aircraft_type",
        "flight_number",
        "flight_arrival",
    ]
    for col in cat_cols:
        X[col] = X[col].astype(str)

    train_pool = Pool(X, y, cat_features=cat_cols)

    hyper_params = {
        "iterations": 800,
        "learning_rate": 0.12,
        "l2_leaf_reg": 15,
        "loss_function": "MAE",
        "thread_count": -1,
        "metric_period": 50,
        "max_depth": 9,
        "max_bin": 63,
        "eval_metric": "MAE",
    }

    if airport == "KMEM":
        hyper_params["iterations"] = 1500
    elif airport == "KJFK":
        hyper_params["iterations"] = 1000
    elif airport == "KMIA":
        hyper_params["iterations"] = 1200

    model = CatBoostRegressor(**hyper_params)
    model.fit(train_pool)
    model.save_model(model_directory / f"catboost_{airport}_final")

    return model


def train_negative_model(neg_pred):
    X = neg_pred.drop(["minutes_until_pushback", "predicted"], axis=1)
    y = neg_pred["minutes_until_pushback"]

    cat_cols = [
        "aircraft_engine_class",
        "major_carrier",
        "aircraft_type",
        "flight_number",
        "flight_arrival",
    ]
    for col in cat_cols:
        X[col] = X[col].astype(str)

    train_pool = Pool(X, y, cat_features=cat_cols)
    hyper_params = {
        "iterations": 800,
        "learning_rate": 0.01,
        "l2_leaf_reg": 20,
        "loss_function": "MAE",
        "thread_count": -1,
        "metric_period": 50,
        "max_depth": 4,
        "max_bin": 63,
        "eval_metric": "MAE",
    }

    model = CatBoostRegressor(**hyper_params)
    model.fit(train_pool)
    model.save_model(model_directory / "catboost_neg_val")


if __name__ == "__main__":
    # Train catboost model on each airport
    for airport in airports:
        # Extract and save dataset
        data_df = make_dataset(airport)

        model_directory.mkdir(parents=True, exist_ok=True)
        train_new_model(data_df)

    # Create model for negative predictions
    neg_pred = get_negative_predictions()
    if len(neg_pred) > 0:
        train_negative_model(neg_pred)
