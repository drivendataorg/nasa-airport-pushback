#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Trevor Tomlin
# - Daniil Filienko
# This script evaluates loaded model on the validation data.
#
import torch
import pickle
import pandas as pd
import numpy as np
import sys
import os

_ROOT: str = os.path.join(os.path.dirname(__file__), "..")

# the path to the directory where data files are stored
DATA_DIR: str = os.path.join(_ROOT, "_data")
ASSETS_DIR: str = os.path.join(_ROOT, "assets")
TRAIN_DIR: str = os.path.join(_ROOT, "training")

sys.path.append(TRAIN_DIR)
from federated import train
from net_1 import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, df):
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Create a new NumPy array with correct numeric data types
    numeric_np_array = numeric_df.values.astype(np.float32)

    # Convert the new NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(numeric_np_array).to(DEVICE)

    # Make predictions using the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(tensor)

    # Convert the predictions back to a pandas DataFrame
    df_output = pd.DataFrame(predictions.cpu().numpy(), columns=["minutes_until_pushback"])

    return df_output


def load_model(assets_directory, num):
    """Load all model assets from disk."""
    model = None
    encoder = None
    with open(os.path.join(_ROOT, "assets", "encoders.pickle"), "rb") as fp:
        encoder = pickle.load(fp)
    with open(os.path.join(_ROOT, "assets", f"model_{num}.pt"), "rb") as fp:
        model = torch.load(fp)
    # with open(assets_directory + "/model.pkl", "rb") as f:
    #    model = pickle.load(f)
    # model.to('cuda')

    return model, encoder


def encode_df(_df: pd.DataFrame, encoded_columns: list, int_columns: list, encoders) -> pd.DataFrame:
    for column in encoded_columns:
        try:
            _df[column] = encoders[column].transform(_df[[column]])
        except Exception as e:
            print(e)
            print(column)
            print(_df.shape)
    for column in int_columns:
        try:
            _df[column] = _df[column].astype("int")
        except Exception as e:
            print(e)
            print(column)
    return _df


if __name__ == "__main__":
    import argparse
    from glob import glob
    from generate import generate
    from utils import *

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-t", help="training the model")
    parser.add_argument("-d", help="directory where inference data is located")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-n", help="model version")
    args: argparse.Namespace = parser.parse_args()

    training: bool = False
    # If have not been run before, run the training.
    if len(glob(os.path.join(ASSETS_DIR, "*"))) == 0 or str(args.t).lower().startswith("t"):
        training = True

    INFERENCE_DATA_DIR: str = os.path.join(_ROOT, "_data") if args.d is None else os.path.join(_ROOT, str(args.d))
    model_version: int = 15 if args.n is None else int(args.n)

    # airports evaluated for
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")

    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    submission_format = pd.read_csv(f"{INFERENCE_DATA_DIR}/submission_format.csv", parse_dates=["timestamp"])

    # Training run
    if training:
        print("Preparing the data, Training")
        generate(airports, _ROOT)
        print("Intermediate Tables were successfully generated")
        print("Training the model")
        train()

    # If have not been run before, run the inference data preprocessing.
    if len(glob(os.path.join(INFERENCE_DATA_DIR, "validation_tables", "*"))) == 0:
        print("Preparing the data, Inference")
        generate(airports, _ROOT, INFERENCE_DATA_DIR, INFERENCE_DATA_DIR, submission_format)

    full_val_table = get_inference_data(INFERENCE_DATA_DIR, airlines, airports)
    model, encoder = load_model(ASSETS_DIR, model_version)
    _df = encode_df(full_val_table, encoded_columns, int_columns, encoder)

    # evaluating the output
    predictions = predict(model, _df[features])

    output_df = _df[["gufi", "timestamp"]]
    output_df["minutes_until_pushback"] = predictions

    print("Finished evaluation")
    print("------------------------------")

    submission_format.drop(columns=["minutes_until_pushback"], inplace=True)
    submission_format.merge(output_df, on=["gufi", "timestamp"]).to_csv("submission.csv", index=False)
