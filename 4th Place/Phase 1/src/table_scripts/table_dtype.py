#
# Author: Yudong Lin
#
# A simple class for keep tracking of typing for output table
#

import pandas as pd


class TableDtype:
    INT_COLUMNS: tuple[str, ...] = (
        "minutes_until_pushback",
        "minutes_until_etd",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
    )

    FLOAT_COLUMNS: tuple[str, ...] = ("cloud_ceiling", "visibility")

    STR_COLUMS: tuple[str, ...] = (
        "cloud",
        "lightning_prob",
        "precip",
        "departure_runway_actual",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "gufi_end_label",
    )

    # fill potential missing int features with 0
    @staticmethod
    def fix_potential_missing_int_features(_df: pd.DataFrame) -> pd.DataFrame:
        columns_need_normalize: tuple[str, ...] = (
            "temperature",
            "wind_direction",
            "wind_speed",
            "wind_gust",
            "cloud_ceiling",
            "visibility",
        )

        for _col in columns_need_normalize:
            if _col in _df.columns:
                _df[_col] = _df[_col].fillna(0)

        return _df
