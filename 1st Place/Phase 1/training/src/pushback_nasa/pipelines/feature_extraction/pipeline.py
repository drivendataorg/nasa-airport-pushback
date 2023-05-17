from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_perimeter,
    extract_mfs_features,
    extract_config_features,
    extract_etd_features,
    extract_moment_features,
    extract_weather_features,
    extract_runway_features,
    extract_standtime_features,
    extract_tfm_features,
    extract_tbfm_features,
)


def create_pipeline(**kwargs) -> Pipeline:
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

    perimeter = [
        node(
            func=compute_perimeter,
            inputs=[f"raw_{airport}_labels", f"sub_format", f"params:{airport}"],
            outputs=f"perimeter_{airport}",
            name=f"computing_perimeter_{airport}",
        )
        for airport in airports
    ]

    mfs_features = [
        node(
            func=extract_mfs_features,
            inputs=[f"perimeter_{airport}", f"raw_{airport}_mfs"],
            outputs=f"features_{airport}_mfs",
            name=f"extract_{airport}_mfs_features",
        )
        for airport in airports
    ]

    etd_features = [
        node(
            func=extract_etd_features,
            inputs=[f"perimeter_{airport}", f"raw_{airport}_etd"],
            outputs=f"features_{airport}_etd",
            name=f"extract_{airport}_etd_features",
        )
        for airport in airports
    ]

    moment_features = [
        node(
            func=extract_moment_features,
            inputs=f"perimeter_{airport}",
            outputs=f"features_{airport}_moment",
            name=f"extract_{airport}_mom_features",
        )
        for airport in airports
    ]

    config_features = [
        node(
            func=extract_config_features,
            inputs=[f"raw_{airport}_config", "params:start_time", "params:end_time"],
            outputs=f"features_{airport}_config",
            name=f"extract_{airport}_config_features",
        )
        for airport in airports
    ]

    weather_features = [
        node(
            func=extract_weather_features,
            inputs=[f"raw_{airport}_lamp", "params:start_time", "params:end_time"],
            outputs=f"features_{airport}_lamp",
            name=f"extract_{airport}_weather_features",
        )
        for airport in airports
    ]

    runway_features = [
        node(
            func=extract_runway_features,
            inputs=[f"raw_{airport}_runways", "params:start_time", "params:end_time"],
            outputs=f"features_{airport}_runways",
            name=f"extract_{airport}_runway",
        )
        for airport in airports
    ]

    standtime_features = [
        node(
            func=extract_standtime_features,
            inputs=[
                f"perimeter_{airport}",
                f"raw_{airport}_standtimes",
                f"raw_{airport}_etd",
            ],
            outputs=f"features_{airport}_standtimes",
            name=f"extract_{airport}_standtimes",
        )
        for airport in airports
    ]

    tfm_features = [
        node(
            func=extract_tfm_features,
            inputs=[f"raw_{airport}_tfm"],
            outputs=f"features_{airport}_tfm",
            name=f"extract_{airport}_tfm",
        )
        for airport in airports
    ]

    tbfm_features = [
        node(
            func=extract_tbfm_features,
            inputs=[f"raw_{airport}_tbfm"],
            outputs=f"features_{airport}_tbfm",
            name=f"extract_{airport}_tbfm",
        )
        for airport in airports
    ]

    return pipeline(
        perimeter
        + mfs_features
        + etd_features
        + moment_features
        + config_features
        + weather_features
        + runway_features
        + standtime_features
    )
