#
# Author: Yudong Lin
#
# A set of useful tools
#

import json
import os
import pickle
from copy import deepcopy
from glob import glob
from typing import Any, Final

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import tensorflow as tf  # type: ignore
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore

from .constants import *

"""
arguments and constants
"""
# the path to root
_ROOT_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "..", "..")
# path to model and encoder
_MODEL_SAVE_TO: Final[str] = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets"
)
# if use one hot encoder
_USE_ONE_HOT: Final[bool] = False

_PRIVATE_CATEGORICAL_STR_COLUMNS: list[str] = [
    "aircraft_engine_class",
    "major_carrier",
    "flight_type",
    "aircraft_type",
]

_CATEGORICAL_STR_COLUMNS: list[str] = [
    "airport",
    "cloud",
    "lightning_prob",
    "precip",
    "departure_runways",
    "arrival_runways",
    "gufi_flight_destination_airport",
    "airline",
    "year",
] + _PRIVATE_CATEGORICAL_STR_COLUMNS

ENCODED_STR_COLUMNS: list[str] = deepcopy(_CATEGORICAL_STR_COLUMNS)


_FEATURES_IGNORE: list[str] = [
    "gufi",
    "timestamp",
    "isdeparture",
    "aircraft_engine_class",
    "precip",
    "departure_runways",
    "arrival_runways",
    "gufi_flight_destination_airport",
    # "minute",
    # "year",
]


def _get_tables(
    _path: str, remove_duplicate_gufi: bool, use_cols: list[str] | None = None
) -> pd.DataFrame:
    unknown_dtype: dict = {
        "precip": str,
        "airline": str,
        "arrival_runways": str,
        "year": str,
    }
    for _col in FLOAT32_COLUMNS:
        unknown_dtype[_col] = "float32"
    for _col in INT16_COLUMNS:
        unknown_dtype[_col] = "int16"
    for _col in CATEGORICAL_INT8_COLUMNS:
        unknown_dtype[_col] = "int8"
    _df: pd.DataFrame
    if "ALL_" not in _path:
        _df = (
            pd.read_csv(_path, parse_dates=["timestamp"], dtype=unknown_dtype)
            if use_cols is None
            else pd.read_csv(_path, dtype=unknown_dtype, usecols=use_cols)
        )
    else:
        _df = pd.concat(
            [
                (
                    pd.read_csv(
                        each_csv_path, parse_dates=["timestamp"], dtype=unknown_dtype
                    )
                    if use_cols is None
                    else pd.read_csv(
                        each_csv_path, dtype=unknown_dtype, usecols=use_cols
                    )
                )
                for each_csv_path in glob(
                    os.path.join(os.path.dirname(os.path.dirname(_path)), "*", "*.csv")
                )
            ],
            ignore_index=True,
        )
    if remove_duplicate_gufi is True:
        _df = _df.drop_duplicates(subset=["gufi"])
    return _df


def get_train_tables_path(_airport: str, _airline: str) -> str:
    return os.path.join(_ROOT_PATH, "train_tables", _airport, f"{_airline}_train.csv")


def get_all_train_tables(airline: str, remove_duplicate_gufi: bool) -> pd.DataFrame:
    return pd.concat(
        [
            _get_tables(pd_path, remove_duplicate_gufi)
            for pd_path in glob(
                os.path.join(_ROOT_PATH, "train_tables", "*", f"{airline}_train.csv")
            )
        ]
        if airline != "ALL"
        else [
            _get_tables(pd_path, remove_duplicate_gufi)
            for pd_path in glob(
                os.path.join(_ROOT_PATH, "train_tables", "*", f"*_train.csv")
            )
            if "PUBLIC_" not in pd_path
        ]
    )


def get_train_tables(
    _airport: str = "KSEA", airline: str = "PUBLIC", remove_duplicate_gufi: bool = True
) -> pd.DataFrame:
    return (
        _get_tables(get_train_tables_path(_airport, airline), remove_duplicate_gufi)
        if _airport != "ALL"
        else get_all_train_tables(airline, remove_duplicate_gufi)
    )


def get_validation_tables_path(_airport: str, _airline: str) -> str:
    return os.path.join(
        _ROOT_PATH, "validation_tables", _airport, f"{_airline}_validation.csv"
    )


def get_all_validation_tables(
    airline: str, remove_duplicate_gufi: bool
) -> pd.DataFrame:
    return pd.concat(
        [
            _get_tables(pd_path, remove_duplicate_gufi)
            for pd_path in glob(
                os.path.join(
                    _ROOT_PATH, "validation_tables", "*", f"{airline}_validation.csv"
                )
            )
        ]
        if airline != "ALL"
        else [
            _get_tables(pd_path, remove_duplicate_gufi)
            for pd_path in glob(
                os.path.join(_ROOT_PATH, "validation_tables", "*", f"*_validation.csv")
            )
            if "PUBLIC_" not in pd_path
        ]
    )


def get_validation_tables(
    _airport: str = "KSEA", airline: str = "PUBLIC", remove_duplicate_gufi: bool = True
) -> pd.DataFrame:
    return (
        _get_tables(
            get_validation_tables_path(_airport, airline), remove_duplicate_gufi
        )
        if _airport != "ALL"
        else get_all_validation_tables(airline, remove_duplicate_gufi)
    )


def get_private_master_tables(
    remove_duplicate_gufi: bool = False, use_cols: list[str] | None = None
) -> pd.DataFrame:
    return pd.concat(
        [
            _get_tables(pd_path, remove_duplicate_gufi, use_cols)
            for pd_path in glob(
                os.path.join(_ROOT_PATH, "full_tables", "*", "*_full.csv")
            )
            if "PUBLIC_" not in pd_path
        ]
    )


def get_model_path(_fileName: str | None) -> str:
    if not os.path.exists(_MODEL_SAVE_TO):
        os.mkdir(_MODEL_SAVE_TO)
    return (
        os.path.join(_MODEL_SAVE_TO, _fileName)
        if _fileName is not None
        else _MODEL_SAVE_TO
    )


def get_clean_categorical_columns() -> list[str]:
    ignore_categorical_features(_FEATURES_IGNORE)
    return ENCODED_STR_COLUMNS + CATEGORICAL_INT8_COLUMNS


def ignore_categorical_features(features_ignore: list[str]) -> None:
    for _ignore in features_ignore:
        if _ignore in ENCODED_STR_COLUMNS:
            ENCODED_STR_COLUMNS.remove(_ignore)
        if _ignore in CATEGORICAL_INT8_COLUMNS:
            CATEGORICAL_INT8_COLUMNS.remove(_ignore)


ignore_categorical_features(_FEATURES_IGNORE)


def get_encoder() -> dict[str, OrdinalEncoder]:
    # generate encoders if not exists
    if os.path.exists(get_model_path("encoders.pickle")):
        with open(get_model_path("encoders.pickle"), "rb") as handle:
            return pickle.load(handle)
    else:
        _encoder: dict[str, OrdinalEncoder] = {}
        print(f"No encoder is found, will generate one right now.")
        # _df: pd.DataFrame = get_private_master_tables(use_cols=_CATEGORICAL_STR_COLUMNS)
        # _encoder["_UNIQUE"] = {}
        # need to make provisions for handling unknown values
        for _col in _CATEGORICAL_STR_COLUMNS:
            """
            _encoder[_col] = (
                OneHotEncoder(
                    categories=CATEGORICAL_STR_CATEGORIES.get(_col, "auto"),
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ).set_output(transform="pandas")
                if _USE_ONE_HOT
                else OrdinalEncoder(
                    categories=CATEGORICAL_STR_CATEGORIES.get(_col, "auto"),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
            ).fit(_df[[_col]])
            """
            _encoder[_col] = (
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ).set_output(transform="pandas")
                if _USE_ONE_HOT
                else OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            ).fit(pd.DataFrame({_col: CATEGORICAL_STR_CATEGORIES[_col]})[[_col]])
            # _encoder["_UNIQUE"][_col] = _df[_col].unique().tolist()
        # print(_encoder["_UNIQUE"])
        # save the encoder
        with open(get_model_path("encoders.pickle"), "wb") as handle:
            pickle.dump(_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return _encoder


def generate_normalization_layer() -> tf.keras.layers.Normalization:
    train_df, val_df = get_train_and_test_ds("ALL", "PRIVATE_ALL")
    normalizer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(axis=-1)
    X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
    X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
    normalizer.adapt(X_test)
    normalizer.adapt(X_train)
    return normalizer


def any_ds_exists(_airline: str) -> bool:
    return (
        len(glob(os.path.join(_ROOT_PATH, "full_tables", "*", f"{_airline}_full.csv")))
        > 0
    )


# get the train and test dataset
def get_train_and_test_ds(
    _airport: str, _airline: str = "PUBLIC"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    if _airline != "PRIVATE_ALL":
        train_df = get_train_tables(_airport, _airline, remove_duplicate_gufi=False)
        val_df = get_validation_tables(_airport, _airline, remove_duplicate_gufi=False)
    elif _airport.upper() != "ALL":
        train_df = pd.concat(
            [
                get_train_tables(_airport, each_airline, remove_duplicate_gufi=False)
                for each_airline in AIRLINES
                if os.path.exists(get_train_tables_path(_airport, each_airline))
            ]
        )
        val_df = pd.concat(
            [
                get_validation_tables(
                    _airport, each_airline, remove_duplicate_gufi=False
                )
                for each_airline in AIRLINES
                if os.path.exists(get_validation_tables_path(_airport, each_airline))
            ]
        )
    else:
        train_df = get_all_train_tables("ALL", remove_duplicate_gufi=False)
        val_df = get_all_validation_tables("ALL", remove_duplicate_gufi=False)

    # load encoder
    _ENCODER: dict[str, OrdinalEncoder | OneHotEncoder] = get_encoder()

    # need to make provisions for handling unknown values
    for col in ENCODED_STR_COLUMNS:
        if col in train_df.columns:
            if isinstance(_ENCODER[col], OrdinalEncoder):
                if len(train_df[col]) > 0:
                    train_df[[col]] = _ENCODER[col].transform(train_df[[col]])
                if len(val_df[col]) > 0:
                    val_df[[col]] = _ENCODER[col].transform(val_df[[col]])
            else:
                train_df = pd.concat(
                    [train_df, _ENCODER[col].transform(train_df[[col]])], axis=1
                ).drop([col], axis=1)
                val_df = pd.concat(
                    [val_df, _ENCODER[col].transform(val_df[[col]])], axis=1
                ).drop([col], axis=1)

    for col in ENCODED_STR_COLUMNS + CATEGORICAL_INT8_COLUMNS:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("int8")
            val_df[col] = val_df[col].astype("int8")

    # drop useless columns
    train_df.drop(
        columns=(_col for _col in _FEATURES_IGNORE if _col in train_df), inplace=True
    )
    val_df.drop(
        columns=(_col for _col in _FEATURES_IGNORE if _col in val_df), inplace=True
    )

    return train_df, val_df


class ModelRecords:
    # create or load model records
    __PATH: str = get_model_path("model_records.json")
    __DATA: dict[str, dict[str, dict]] = {}
    __init: bool = False

    @classmethod
    def init(cls) -> None:
        if os.path.exists(cls.__PATH):
            with open(cls.__PATH, "r", encoding="utf-8") as f:
                cls.__DATA = dict(json.load(f))
        cls.__init = True

    @classmethod
    def set_name(cls, fileName: str) -> None:
        cls.__PATH = get_model_path(fileName + ".json")
        cls.__init = False

    @classmethod
    def get(cls, _airport: str) -> dict[str, dict]:
        if not cls.__init:
            cls.init()
        if _airport not in cls.__DATA:
            cls.__DATA[_airport] = {}
        return cls.__DATA[_airport]

    @classmethod
    def get_smallest(cls, _airport: str, _key: str = "val_mae") -> dict | None:
        results: dict[str, dict] = cls.get(_airport)
        if len(results) == 0:
            return None
        _best_result: dict = {_key: 999999999999}
        for _recode in results.values():
            if float(_recode[_key]) < float(_best_result[_key]):
                _best_result = _recode
        return _best_result

    @classmethod
    def update(cls, _airport: str, _key: str, _value: Any, save: bool = False) -> None:
        cls.get(_airport)[_key] = _value
        if save is True:
            cls.save()

    @classmethod
    def save(cls) -> None:
        with open(cls.__PATH, "w", encoding="utf-8") as f:
            json.dump(cls.__DATA, f, indent=4, ensure_ascii=False, sort_keys=True)

    @classmethod
    def display_best(cls) -> None:
        for _airport in ALL_AIRPORTS:
            _best: dict | None = cls.get_smallest(_airport)
            if _best is not None:
                print("--------------------")
                print(f"{_airport}:")
                print("train mae:", _best["train_mae"])
                print("val mae:", _best["val_mae"])


def plot_history(
    _airport: str, history: dict[str, list], saveAsFileName: str | None = None
) -> None:
    plt.clf()
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"validation loss curve for {_airport}")
    if saveAsFileName is not None:
        plt.savefig(get_model_path(saveAsFileName))
