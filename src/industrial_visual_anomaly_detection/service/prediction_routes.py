from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import UnidentifiedImageError

from .heatmap_encoding import encode_heatmap_png_base64
from .prediction_response import HeatmapResponse, PredictionResponse
from .runtime import InferenceRuntime

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["predictions"],
)


@router.post("", response_model=PredictionResponse)
def create_prediction(
    request: Request,
    image: UploadFile = File(...),
) -> PredictionResponse:
    """Analyze one uploaded image using the loaded model runtime."""

    runtime: InferenceRuntime = request.app.state.inference_runtime

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
        output_size=(runtime.input_size, runtime.input_size),
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