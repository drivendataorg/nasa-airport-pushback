from features import *
import argparse, time, os, sys, pickle
import pandas as pd
import numpy as np
import xgboost

# USING STORED FEATURES
FINAL_COLS = ["gufi", "timestamp", "airport", "minutes_until_pushback"]

MODEL_PATH = "/models/v5_model_{}_{}.json"
DATA_PATH = "/Users/cuontr/Documents/pushback/data/{}/{}/"
LABEL_PATH = "/Users/cuontr/Documents/pushback/data/{}/"
SAVE_PATH = "./data/processed/submission_{}_{}.csv"


if "cuontr" in os.getcwd():
    prediction_pd = pd.read_csv(
        "/Users/cuontr/Documents/pushback/data/submission_format.csv",
        parse_dates=["timestamp"],
    )


def run_inference(airport_name, lr):

    np.random.seed(0)
    sub_file_name = SAVE_PATH.format(airport_name, lr)
    model_file_name = MODEL_PATH.format(airport_name, lr)
    submission_pd = prediction_pd[prediction_pd["airport"] == airport_name]
    # EXTRACT FEATURES FOR TRAINING DATA
    features = []
    cat_features = []
    etd_file_name = DATA_PATH.format(
        airport_name, airport_name
    ) + "{}_etd.csv.bz2".format(airport_name)
    etd = pd.read_csv(
        etd_file_name, parse_dates=["departure_runway_estimated_time", "timestamp"]
    )

    rw_file_name = DATA_PATH.format(
        airport_name, airport_name
    ) + "{}_runways.csv.bz2".format(airport_name)
    runways = pd.read_csv(rw_file_name, parse_dates=["timestamp"])

    weather_file_name = DATA_PATH.format(
        airport_name, airport_name
    ) + "{}_lamp.csv.bz2".format(airport_name)
    lamp = pd.read_csv(
        weather_file_name, parse_dates=["forecast_timestamp"]
    ).drop_duplicates(subset="forecast_timestamp")

    meta_file_name = DATA_PATH.format(
        airport_name, airport_name
    ) + "{}_mfs.csv.bz2".format(airport_name)
    mfs = pd.read_csv(meta_file_name)

    submission_pd, feat_dict = get_diff_depart_vs_ts(submission_pd, etd)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]
    submission_pd, feat_dict = get_weather_info(submission_pd, lamp, airport_name)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]
    submission_pd, feat_dict = get_basic_feats(submission_pd)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]
    submission_pd, feat_dict = get_meta_features(submission_pd, mfs)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]
    submission_pd, feat_dict = get_dep_rw(submission_pd, runways)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]
    submission_pd, feat_dict = get_num_dep_flights(submission_pd, runways)
    features += feat_dict["numeric"] + feat_dict["cat"]
    cat_features += feat_dict["cat"]

    values = {v: -1 for v in features if v not in cat_features}
    for v in cat_features:
        submission_pd[v] = submission_pd[v].astype("string")
        submission_pd[v].fillna("NA", inplace=True)
        submission_pd[v] = submission_pd[v].astype("category")
    submission_pd = submission_pd.fillna(values)
    submission_pd = pd.get_dummies(submission_pd, columns=cat_features)

    model = xgboost.XGBRegressor()
    model.load_model(model_file_name)
    old_features = model.get_booster().feature_names
    for x in old_features:
        if x not in submission_pd.columns.tolist():
            submission_pd[x] = 0

    y_pred = model.predict(submission_pd[old_features])
    submission_pd["minutes_until_pushback"] = y_pred
    submission_pd["minutes_until_pushback"] = submission_pd[
        "minutes_until_pushback"
    ].apply(lambda x: max(int(x), 1))

    submission_pd[FINAL_COLS].to_csv(sub_file_name, index=False)


def main():
    starttime = time.time()
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--airport_name", default="KATL", type=str)
    parser.add_argument("--lr", default=2.5, type=float)
    args = parser.parse_args()
    run_inference(args.airport_name, args.lr)
    print("That took {} seconds".format(time.time() - starttime))


if __name__ == "__main__":
    main()
