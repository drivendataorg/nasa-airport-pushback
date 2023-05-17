import pandas as pd


def adjust_types(master: pd.DataFrame) -> pd.DataFrame:
    # NaN imputation method
    master = master.ffill().bfill().fillna(0)

    object_cols = [
        "mfs_cat_flightid",
        "mfs_cat_airline",
        "mfs_cat_flightno",
        "mfs_cat_no_last",
        "mfs_cat_arrival",
        "mfs_cat_last",
        "mfs_cat_sectolast",
        "mfs_cat_engineclass",
        "mfs_cat_type",
        "mfs_cat_majorcarrier",
        "mfs_cat_flighttype",
    ]

    # Variable type change to make the master table more lightweight
    for c in master.columns:
        if c not in (["gufi", "timestamp", "minutes_until_pushback"] + object_cols):
            master[c] = master[c].astype("int16")

    return master


def build_master(
    base: pd.DataFrame,
    mfs: pd.DataFrame,
    config: pd.DataFrame,
    etd: pd.DataFrame,
    mom: pd.DataFrame,
    weather: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tfm: pd.DataFrame,
    tbfm: pd.DataFrame,
) -> pd.DataFrame:
    # Select columns from base
    base = base[["gufi", "timestamp", "minutes_until_pushback"]]

    # Merge all feature blocks in a single master table
    master = base.merge(mfs, how="left", on=["gufi", "timestamp"])
    master = master.merge(config, how="left", on="timestamp")
    master = master.merge(etd, how="left", on=["gufi", "timestamp"])
    master = master.merge(mom, how="left", on="timestamp")
    master = master.merge(weather, how="left", on="timestamp")
    master = master.merge(runways, how="left", on="timestamp")
    master = master.merge(standtimes, how="left", on="timestamp")
    # master = master.merge(tfm, how="left", on="timestamp")
    # master = master.merge(tbfm, how="left", on="timestamp")
    master = adjust_types(master)

    return master


def build_global_master(*args):
    global_master = pd.DataFrame()

    airports = [
        "katl",
        "kclt",
        "kden",
        "kdfw",
        "kjfk",
        "kmem",
        "kmia",
        "kord",
        "kphx",
        "ksea",
    ]

    for i in range(10):
        table = args[i]
        table = table[
            ["gufi", "timestamp", "minutes_until_pushback"]
            + [c for c in table.columns if ("mfs" in c) or ("etd" in c)]
        ]
        table["feat_cat_airport"] = airports[i]
        global_master = pd.concat([global_master, table])

    global_master = global_master.sort_values("timestamp").reset_index(drop=True)
    return global_master
