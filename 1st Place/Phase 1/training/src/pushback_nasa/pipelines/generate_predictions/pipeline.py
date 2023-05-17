"""
This is a boilerplate pipeline 'generate_predictions'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import retrieve_predictions, combine_predictions


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

    return pipeline(
        [
            node(
                func=retrieve_predictions,
                inputs=[
                    "sub_format",
                    f"master_{airport}",
                    f"{airport}_model",
                    f"params:{airport}",
                ],
                outputs=f"predictions_{airport}",
                name=f"extract_predictions_{airport}",
            )
            for airport in airports
        ]
        + [
            node(
                func=combine_predictions,
                inputs=[f"predictions_{airport}" for airport in airports],
                outputs="submission",
                name="combine_submissions",
            )
        ]
    )
