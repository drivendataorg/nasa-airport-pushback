import flwr
import pandas as pd
import tensorflow as tf  # type: ignore

from .constants import TARGET_LABEL


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, train_df: pd.DataFrame, val_df: pd.DataFrame):
        super().__init__()

        self.__X_train: tf.Tensor = tf.convert_to_tensor(
            train_df.drop(columns=[TARGET_LABEL])
        )
        self.__X_test: tf.Tensor = tf.convert_to_tensor(
            val_df.drop(columns=[TARGET_LABEL])
        )

        self.__y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL])
        self.__y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL])

        self.__model = model

    def get_parameters(self, config):
        return self.__model.get_weights()

    def fit(self, parameters, config):
        self.__model.set_weights(parameters)
        self.__model.fit(
            self.__X_train, self.__y_train, epochs=1, batch_size=32, steps_per_epoch=3
        )
        return self.__model.get_weights(), len(self.__X_train), {}

    def evaluate(self, parameters, config):
        self.__model.set_weights(parameters)
        loss = self.__model.evaluate(self.__X_test, self.__y_test)
        return loss, len(self.__X_test), {"loss": loss}
