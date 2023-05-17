#
# Author: Yudong Lin
#
# extract date based features from the timestamp
#

import pandas as pd


def add_date_features(_df: pd.DataFrame) -> pd.DataFrame:
    from pandarallel import pandarallel  # type: ignore

    pandarallel.initialize(verbose=1)

    _df["year"] = _df.parallel_apply(lambda x: x.timestamp.year, axis=1)
    _df["month"] = _df.parallel_apply(lambda x: x.timestamp.month, axis=1)
    _df["day"] = _df.parallel_apply(lambda x: x.timestamp.day, axis=1)
    _df["hour"] = _df.parallel_apply(lambda x: x.timestamp.hour, axis=1)
    _df["minute"] = _df.parallel_apply(lambda x: x.timestamp.minute, axis=1)
    _df["weekday"] = _df.parallel_apply(lambda x: x.timestamp.weekday(), axis=1)

    return _df
