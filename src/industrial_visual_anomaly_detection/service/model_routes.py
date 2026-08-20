from fastapi import APIRouter, Request

from .model_response import (
    ModelCatalogResponse,
    ModelResponse,
)
from .runtime import InferenceRuntime
from .runtime_registry import InferenceRuntimeRegistry

router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
)


@router.get("", response_model=ModelCatalogResponse)
def get_models(
    request: Request,
) -> ModelCatalogResponse:
    """Return all models available for inference."""

    runtime_source: (
        InferenceRuntime | InferenceRuntimeRegistry
    ) = request.app.state.inference_runtime

    if isinstance(
        runtime_source,
        InferenceRuntimeRegistry,
    ):
        return ModelCatalogResponse(
            defaultModelId=runtime_source.default_model_id,
            models=[
                ModelResponse(
                    id=model.model_id,
                    displayName=model.display_name,
                    category=model.category,
                    inputSize=model.input_size,
                    isDefault=model.is_default,
                )
                for model in runtime_source.available_models
            ],
        )

    return ModelCatalogResponse(
        defaultModelId=runtime_source.model_id,
        models=[
            ModelResponse(
                id=runtime_source.model_id,
                displayName=(
                    runtime_source.category
                    .replace("_", " ")
                    .title()
                ),
                category=runtime_source.category,
                inputSize=runtime_source.input_size,
                isDefault=True,
            )
        ],
    )