#
# Author: Yudong Lin
#
# extract potential useful features from gufi
#

import pandas as pd


def extract_and_add_gufi_features(_df: pd.DataFrame) -> pd.DataFrame:
    from pandarallel import pandarallel  # type: ignore

    pandarallel.initialize(verbose=1)

    def _split_gufi(x: pd.DataFrame) -> pd.Series:
        from datetime import datetime

        information: list = x["gufi"].split(".")
        gufi_flight_number: str = information[0]
        gufi_flight_major_carrier: str = gufi_flight_number[:3]
        gufi_flight_destination_airport: str = information[2]
        gufi_flight_date: datetime = datetime.strptime(
            "_".join((information[3], information[4], information[5][:2])), "%y%m%d_%H%M_%S"
        )
        # gufi_flight_FAA_system: str = information[6]
        gufi_timestamp_until_etd = int((gufi_flight_date - x["timestamp"]).seconds / 60)
        return pd.Series(
            [
                # gufi_flight_number,
                gufi_flight_major_carrier,
                gufi_flight_destination_airport,
                gufi_timestamp_until_etd,
                # gufi_flight_date,
                # gufi_flight_FAA_system,
            ]
        )

    _df[
        [
            # "gufi_flight_number",
            "airline",
            "gufi_flight_destination_airport",
            "gufi_timestamp_until_etd",
            # "gufi_flight_date",
            # "gufi_flight_FAA_system",
        ]
    ] = _df.parallel_apply(lambda x: _split_gufi(x), axis=1)

    return _df
