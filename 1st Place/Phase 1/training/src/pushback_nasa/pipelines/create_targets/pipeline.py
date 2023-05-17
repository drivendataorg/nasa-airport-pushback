"""
This is a boilerplate pipeline 'create_targets'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_targets


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

    targets = [
        node(
            func=create_targets,
            inputs=[
                f"raw_{airport}_standtimes",
                f"raw_{airport}_mfs",
                "params:start_time",
                "params:end_time",
            ],
            outputs=f"targets_{airport}",
            name=f"create_targets_{airport}",
        )
        for airport in airports
    ]

    return pipeline(targets)
