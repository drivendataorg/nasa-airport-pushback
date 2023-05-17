"""Centralized version of federated evaluation that does not use Flower."""
import numpy as np
import pandas as pd
from loguru import logger
import pickle
from config import (
    AIRLINES,
    AIRPORTS,
    data_directory_raw,
    ann_model_directory,
    xgb_model_directory,
    curr_xgb_model,
    predictions_dir,
)
from utils import (
    load_private_airline_airport_features,
    load_public_airport_features,
    initialize_model,
    private_data_model_specific_preprocessing,
)
from utils import initialize_model
import datetime

if __name__ == "__main__":
    ann_model = initialize_model(n_features=144)
    fp = f"{ann_model_directory}/federated_weights.npz"
    weights = np.load(fp)
    weights = [weights[key] for key in weights]
    ann_model.set_weights(weights)

    predictions = []
    submission_format = pd.read_csv(
        f"{data_directory_raw}/submission_data.csv",
        parse_dates=["timestamp"],
        index_col=["gufi", "timestamp", "airport"],
    )
    submission_format["airline"] = submission_format.index.get_level_values("gufi").str[
        :3
    ]
    reset_sub_format = submission_format.reset_index()
    merged_df = reset_sub_format
    pred_df = None
    AIRPORTS = ["KATL"]
    for airport in AIRPORTS:
        with open(
            f"{xgb_model_directory}/{curr_xgb_model}/{airport}_xgb.pkl", "rb"
        ) as f:
            curr_model = pickle.load(f)
            features_list = curr_model["features"]
            curr_xgb = curr_model["model"]
        print(f"Predicting {airport}")
        for airline in AIRLINES:
            print(f"Predicting {airline} @ {airport}")
            partial_submission_format = (
                submission_format.xs(airport, level="airport", drop_level=False)
                .loc[submission_format.airline == airline]
                .copy()
            )
            if len(partial_submission_format) != 0:
                # try:
                public_features = load_public_airport_features(
                    partial_submission_format.index, airport
                )
                features_df = public_features.reset_index()
                features_df["unix_time"] = (
                    features_df["timestamp"].astype(np.int64) // 10 ** 9
                )
                X_internal_features = features_df[
                    list(curr_xgb.get_booster().feature_names)
                ]
                features_df["predicted_minutes_until_pushback_xgb"] = curr_xgb.predict(
                    X_internal_features
                )

                private_features = load_private_airline_airport_features(
                    partial_submission_format.index, airline, airport
                )

                private_features = private_features.reset_index()
                public_features = public_features.reset_index()
                airline_labels = partial_submission_format.reset_index()
                airline_features = pd.merge(
                    private_features, public_features, on=["gufi", "timestamp"]
                )
                airline_labels = airline_features.merge(
                    airline_labels, on=["gufi", "timestamp"]
                )

                features, labels = private_data_model_specific_preprocessing(
                    airline_features, airline_labels, np.float32
                )

                features_df["predicted_minutes_until_pushback_ann"] = ann_model.predict(
                    features
                )
                features_df["airport"] = airport
                if pred_df is None:
                    pred_df = features_df
                else:
                    pred_df = pd.concat([pred_df, features_df], axis=0)
            # except:
            #    print(f"ERRORED ON {airport} & {airline}, skipping {len(partial_submission_format)} predictions")
    # pred_df[["gufi", "timestamp","predicted_minutes_until_pushback"]] need to only save this
    # Basic Aggregation Strategy
    min_to_pb_df = features_df[
        ["predicted_minutes_until_pushback_xgb", "predicted_minutes_until_pushback_ann"]
    ]
    xgb_weight = 0.8
    ann_weight = 0.2
    assert xgb_weight + ann_weight == 1

    pred_df["final_predict"] = (
        xgb_weight * pred_df["predicted_minutes_until_pushback_xgb"]
    ) + (ann_weight * pred_df["predicted_minutes_until_pushback_ann"])

    presentDate = datetime.datetime.now()
    save_pred_dir_name = str(int(presentDate.timestamp()))
    save_pred_path = f"{predictions_dir}/{save_pred_dir_name}_predictions.csv"
    feats_to_save = [
        "gufi",
        "timestamp",
        "airport",
        "predicted_minutes_until_pushback_xgb",
        "predicted_minutes_until_pushback_ann",
        "final_predict",
    ]
    pred_df = pred_df[feats_to_save]
    pred_df.to_csv(f"{save_pred_path}", index=False)
