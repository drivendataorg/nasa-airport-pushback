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
        "deps_3hr",
        "deps_30hr",
        "arrs_3hr",
        "arrs_30hr",
        "deps_taxiing",
        "arrs_taxiing",
        "exp_deps_15min",
        "exp_deps_30min",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "gufi_timestamp_until_etd",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "weekday",
        "feat_5_gufi",
        "feat_5_estdep_next_30min",
        "feat_5_estdep_next_60min",
        "feat_5_estdep_next_180min",
        "feat_5_estdep_next_360min",
    )

    FLOAT_COLUMNS: tuple[str, ...] = (
        "delay_30hr",
        "standtime_30hr",
        "dep_taxi_30hr",
        "arr_taxi_30hr",
        "delay_3hr",
        "standtime_3hr",
        "dep_taxi_3hr",
        "arr_taxi_3hr",
        "1h_ETDP",
    )

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
    @classmethod
    def fix_potential_missing_int_features(cls, _df: pd.DataFrame) -> pd.DataFrame:
        for _col in cls.INT_COLUMNS:
            if _col in _df.columns:
                _df[_col] = _df[_col].fillna(0).astype("int16")
        for _col in cls.FLOAT_COLUMNS:
            if _col in _df.columns:
                _df[_col] = _df[_col].fillna(0.0).astype("float32")

        return _df
