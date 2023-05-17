from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_catboost_diff, train_catboost


def create_pipeline(**kwargs) -> Pipeline:
    airports = [
        "katl",
        "kclt",
        "kden",
        "kord",
        "kphx",
        "ksea",
        "kmia",
        "kmem",
        "kjfk",
        "kdfw",
    ]

    models_v02 = [
        node(
            func=train_catboost_diff,
            inputs=[f"master_{airport}", "params:target_name", "params:end_train"],
            outputs=[
                f"{airport}_model_v0",
                f"{airport}_model_v2",
                f"{airport}_featimps",
            ],
        )
        for airport in airports
    ]

    models_v1 = [
        node(
            func=train_catboost,
            inputs=[f"master_{airport}", "params:target_name", "params:end_train"],
            outputs=f"{airport}_model_v1",
        )
        for airport in airports
    ]

    global_model = [
        node(
            func=train_catboost,
            inputs=["global_master", "params:target_name", "params:end_train"],
            outputs="global_model",
        )
    ]

    return pipeline(models_v02 + models_v1 + global_model)
