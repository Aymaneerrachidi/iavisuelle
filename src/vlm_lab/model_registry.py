from .domain import ModelInfo, Precision

MODELS: dict[str, ModelInfo] = {
    "blip-base": ModelInfo(
        id="blip-base",
        label="BLIP Base",
        hub_id="Salesforce/blip-image-captioning-base",
        architecture="blip",
        parameters_millions=247,
        supported_precisions=[Precision.FULL, Precision.INT8, Precision.INT4],
        license="BSD-3-Clause",
    ),
    "git-base": ModelInfo(
        id="git-base",
        label="GIT Base",
        hub_id="microsoft/git-base",
        architecture="git",
        parameters_millions=177,
        supported_precisions=[Precision.FULL, Precision.INT8, Precision.INT4],
        license="MIT",
    ),
}


def get_model(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_id}") from exc
