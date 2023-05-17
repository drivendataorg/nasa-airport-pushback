import os
import pickle
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder  # type: ignore


def main() -> None:
    # DATA_DIR = Path("/home/ydlin/FL_Huskies_PTTF/train_tables/")

    airports = [
        "KATL",
        "KCLT",
        "KDEN",
        "KDFW",
        "KJFK",
        "KMEM",
        "KMIA",
        "KORD",
        "KPHX",
        "KSEA",
    ]

    encoded_columns = [
        "airport",
        "departure_runways",
        "arrival_runways",
        "cloud",
        "lightning_prob",
        "precip",
        "gufi_flight_major_carrier",
        "gufi_flight_destination_airport",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
    ]

    unique = defaultdict(set)

    Encoder = partial(
        OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoders = defaultdict(Encoder)

    for airport in airports:
        print(f"Processing Airport {airport}")
        data = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "full_tables", f"{airport}_full.csv"
            )
        ).sort_values("timestamp")
        data["precip"] = data["precip"].astype(str)
        data["isdeparture"] = data["isdeparture"].astype(str)

        for column in encoded_columns:
            unique[column].update(data[column].unique())

    for column in encoded_columns:
        encoders[column].fit(np.array(list(unique[column])).reshape(-1, 1))

    with open(os.path.join(os.path.dirname(__file__), "encoders.pickle"), "wb") as f:
        pickle.dump(encoders, f)


if __name__ == "__main__":
    main()
