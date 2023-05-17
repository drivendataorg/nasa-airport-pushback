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
        import re

        information: list = x["gufi"].split(".")
        gufi_flight_number: str = information[0]
        first_int = re.search(r"\d", gufi_flight_number)
        gufi_flight_major_carrier: str = gufi_flight_number[
            : first_int.start() if first_int is not None else 3
        ]
        gufi_flight_destination_airport: str = information[2]
        return pd.Series([gufi_flight_major_carrier, gufi_flight_destination_airport])

    _df[
        ["gufi_flight_major_carrier", "gufi_flight_destination_airport"]
    ] = _df.parallel_apply(lambda x: _split_gufi(x), axis=1)

    return _df
