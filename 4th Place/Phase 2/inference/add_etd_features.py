#
# Author: Daniil Filienko (?)
#
# Add global lamp data
#

import pandas as pd


def add_etd_features(_df: pd.DataFrame, etd: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts estimated time of departure features and appends it to the existing dataframe
    :param pd.DataFrame _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    etd["timestamp"] = etd.timestamp.dt.ceil("15min")
    etd["departure_runway_estimated_time"] = pd.to_datetime(etd["departure_runway_estimated_time"])
    etd = etd[etd["timestamp"] < etd["departure_runway_estimated_time"]]

    time_intervals: tuple[int, ...] = tuple(sorted([30, 60, 180, 360]))
    increment_in_minutes: int = 15

    complete_etd = pd.DataFrame()
    complete_etd[["gufi", "timestamp", "departure_runway_estimated_time"]] = etd[
        ["gufi", "timestamp", "departure_runway_estimated_time"]
    ]

    for i in range(1, time_intervals[len(time_intervals) - 1] // increment_in_minutes + 2):
        _current = pd.DataFrame()
        _current["timestamp"] = etd["timestamp"] + pd.Timedelta(minutes=i * increment_in_minutes)
        _current[["gufi", "departure_runway_estimated_time"]] = etd[["gufi", "departure_runway_estimated_time"]]
        complete_etd = pd.concat(
            [complete_etd, _current[_current["timestamp"] < _current["departure_runway_estimated_time"]]]
        )

    complete_etd["time_ahead"] = (
        complete_etd["departure_runway_estimated_time"] - complete_etd["timestamp"]
    ).dt.total_seconds()
    complete_etd = complete_etd.groupby(["gufi", "timestamp"]).first().reset_index()

    for _interval in time_intervals:
        complete_etd[f"estdep_next_{_interval}min"] = (complete_etd["time_ahead"] < _interval * 60).astype(int)
    complete_etd.sort_values("time_ahead", inplace=True)

    # for i in [30, 60, 180, 1400]:
    #    complete_etd[f"estdep_num_next_{i}min"] = (complete_etd["time_ahead"] < i * 60).astype(int)
    # complete_etd.sort_values("time_ahead", inplace=True)

    # number of flights departing from the airport
    etd_aggregation = (
        complete_etd.groupby("timestamp")
        .agg(
            {
                "gufi": "count",
                "estdep_next_30min": "sum",
                "estdep_next_60min": "sum",
                "estdep_next_180min": "sum",
                "estdep_next_360min": "sum",
            }
        )
        .reset_index()
    )

    _df = _df.reset_index(drop=True).merge(
        etd_aggregation.rename(columns={k: f"feat_5_{k}" for k in etd_aggregation.columns if k != "timestamp"}),
        how="left",
        on="timestamp",
    )

    return _df
