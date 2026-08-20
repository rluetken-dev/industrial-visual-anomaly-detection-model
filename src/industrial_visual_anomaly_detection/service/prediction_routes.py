from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import UnidentifiedImageError

from .heatmap_encoding import encode_heatmap_png_base64
from .prediction_response import (
    HeatmapResponse,
    PredictionResponse,
)
from .runtime import InferenceRuntime
from .runtime_registry import (
    InferenceRuntimeRegistry,
    UnknownModelError,
)

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["predictions"],
)


@router.post("", response_model=PredictionResponse)
def create_prediction(
    request: Request,
    image: UploadFile = File(...),
    model_id: str | None = Form(
        default=None,
        alias="modelId",
    ),
) -> PredictionResponse:
    """Analyze one uploaded image using the selected model."""

    runtime_source: (
        InferenceRuntime | InferenceRuntimeRegistry
    ) = request.app.state.inference_runtime

    try:
        runtime = _select_runtime(
            runtime_source,
            model_id,
        )
    except UnknownModelError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    try:
        prediction = runtime.predict(image.file)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image.",
        ) from error

    heatmap_data = encode_heatmap_png_base64(
        patch_scores=prediction.patch_scores,
        threshold=prediction.threshold,
        output_size=(
            runtime.input_size,
            runtime.input_size,
        ),
    )

    return PredictionResponse(
        modelId=runtime.model_id,
        category=runtime.category,
        score=prediction.anomaly_score,
        threshold=prediction.threshold,
        isAnomalous=prediction.is_anomalous,
        heatmap=HeatmapResponse(
            contentType="image/png",
            width=runtime.input_size,
            height=runtime.input_size,
            dataBase64=heatmap_data,
        ),
    )


def _select_runtime(
    runtime_source: (
        InferenceRuntime | InferenceRuntimeRegistry
    ),
    model_id: str | None,
) -> InferenceRuntime:
    if isinstance(
        runtime_source,
        InferenceRuntimeRegistry,
    ):
        return runtime_source.get_runtime(model_id)

    if model_id is None:
        return runtime_source

    selected_model_id = model_id.strip()

    if not selected_model_id:
        raise UnknownModelError(
            "Requested model ID must not be empty."
        )

    if selected_model_id != runtime_source.model_id:
        raise UnknownModelError(
            f"Unknown model ID: {selected_model_id}"
        )

    return runtime_source