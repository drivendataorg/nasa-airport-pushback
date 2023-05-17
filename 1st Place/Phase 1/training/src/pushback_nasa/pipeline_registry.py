"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = (
        pipelines["create_targets"]
        + pipelines["feature_extraction"]
        + pipelines["create_master"]
        + pipelines["train_models"]
        + pipelines["generate_predictions"]
    )

    return pipelines
