import torch
import os

ROOT: str = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR: str = os.path.join(ROOT, "_data")
ASSETS_DIR: str = os.path.join(ROOT, "assets")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

AIRPORTS = [
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


AIRLINES = [
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
]

features = [
    "airline",
    "airport",
    "deps_3hr",
    "deps_30hr",
    "deps_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    "1h_ETDP",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "weekday",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "cloud",
    "lightning_prob",
    "gufi_timestamp_until_etd",
]

encoded_columns = [
    "cloud",
    "lightning_prob",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "airport",
    "airline",
]
