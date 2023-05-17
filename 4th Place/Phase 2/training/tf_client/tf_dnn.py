#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import os

from .mytools import *
import tensorflow as tf  # type: ignore

from .constants import ALL_AIRPORTS, TARGET_LABEL


class MyTensorflowDNN:
    DEV_MODE: bool = False
    FEDERATED_MODE: bool = True

    @classmethod
    def create_clean_model(cls) -> None:
        # temporary disable FEDERATED_MODE
        cls.FEDERATED_MODE = False
        model = cls.get_model("ALL")
        # now enable FEDERATED_MODE
        cls.FEDERATED_MODE = True
        model.save(cls.get_model_path(""))

    @classmethod
    def get_model_path(cls, _airport: str) -> str:
        return get_model_path(
            "tf_dnn_model" if cls.FEDERATED_MODE else f"tf_dnn_{_airport}_model"
        )

    @classmethod
    def get_model(
        cls, _airport: str, load_if_exists: bool = True
    ) -> tf.keras.models.Sequential:
        _model: tf.keras.models.Sequential
        model_path: str = cls.get_model_path(_airport)
        if load_if_exists is False or not os.path.exists(model_path):
            # under FEDERATED_MODE, a clean model will be distribute across
            if cls.FEDERATED_MODE is True:
                raise FileNotFoundError("The clean model should is missing!")
            print("----------------------------------------")
            print("Creating new model.")
            print("----------------------------------------")
            _layers: list[tf.keras.layers.Dense] = [
                generate_normalization_layer(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
            _model = tf.keras.models.Sequential(_layers)
            _model.compile(
                loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam()
            )
        else:
            if cls.DEV_MODE:
                print("----------------------------------------")
                print("A existing model has been found and will be loaded.")
                print("----------------------------------------")
            _model = tf.keras.models.load_model(model_path)

        return _model

    @classmethod
    def train(
        cls, _airport: str, load_if_exists: bool = True
    ) -> tf.keras.models.Sequential:
        # load model
        model: tf.keras.models.Sequential = cls.get_model(_airport, load_if_exists)

        # load train and test data frame
        train_df, val_df = get_train_and_test_ds(_airport, "PRIVATE_ALL")

        X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
        X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
        y_train: tf.Tensor = tf.convert_to_tensor(
            train_df[TARGET_LABEL], dtype=tf.int16
        )
        y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

        # show model info
        if cls.DEV_MODE is True:
            model.summary()

        # Model Checkpoint
        check_pointer: tf.keras.callbacks.ModelCheckpoint = (
            tf.keras.callbacks.ModelCheckpoint(
                cls.get_model_path(_airport),
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
            )
        )
        # Model Early Stopping Rules
        early_stopping: tf.keras.callbacks.EarlyStopping = (
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        )

        result = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            verbose=1,
            epochs=50,
            callbacks=[check_pointer, early_stopping],
            batch_size=32 * 8,
        )

        # save history
        if cls.DEV_MODE is True:
            # show params
            print(result.params)
            # update database name
            ModelRecords.set_name("tf_dnn_model_records")
            # save loss history image
            plot_history(_airport, result.history, f"tf_dnn_{_airport}_info.png")
            # save loss history as json
            ModelRecords.update(_airport, "history", result.history, True)

        return model

    @classmethod
    def evaluate_global(cls) -> None:
        _model = tf.keras.models.load_model(cls.get_model_path("ALL"))

        for theAirport in ALL_AIRPORTS:
            # load train and test data frame
            train_df, val_df = get_train_and_test_ds(theAirport)

            X_train: tf.Tensor = tf.convert_to_tensor(
                train_df.drop(columns=[TARGET_LABEL])
            )
            X_test: tf.Tensor = tf.convert_to_tensor(
                val_df.drop(columns=[TARGET_LABEL])
            )
            y_train: tf.Tensor = tf.convert_to_tensor(
                train_df[TARGET_LABEL], dtype=tf.int16
            )
            y_test: tf.Tensor = tf.convert_to_tensor(
                val_df[TARGET_LABEL], dtype=tf.int16
            )

            print(theAirport, ":")
            # _model.evaluate(X_train, y_train)
            _model.evaluate(X_test, y_test)
