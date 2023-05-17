from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_master, build_global_master


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

    master_tables = [
        node(
            func=build_master,
            inputs=[
                f"perimeter_{airport}",
                f"features_{airport}_mfs",
                f"features_{airport}_config",
                f"features_{airport}_etd",
                f"features_{airport}_moment",
                f"features_{airport}_lamp",
                f"features_{airport}_runways",
                f"features_{airport}_standtimes",
                f"features_{airport}_tfm",
                f"features_{airport}_tbfm",
            ],
            outputs=f"master_{airport}",
            name=f"master_{airport}",
        )
        for airport in airports
    ]

    global_master = [
        node(
            func=build_global_master,
            inputs=[f"master_{airport}" for airport in airports],
            outputs="global_master",
            name=f"build_global_master",
        )
    ]

    return pipeline(master_tables + global_master)
