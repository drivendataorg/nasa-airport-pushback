from typing import Any, Dict

import numpy as np

from kedro.io import AbstractDataSet
from catboost import CatBoostRegressor


class CatBoostModel(AbstractDataSet[np.ndarray, np.ndarray]):
    """ CatBoostModel

    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data at the given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        self._filepath = filepath

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array.
        """
        model = CatBoostRegressor()
        model.load_model(self._filepath)
        return model

    def _save(self, model) -> None:
        """Saves image data to the specified filepath"""
        model.save_model(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
