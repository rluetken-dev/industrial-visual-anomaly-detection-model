from fastapi import APIRouter, File, Request, UploadFile

from .prediction_response import PredictionResponse
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
    prediction = runtime.predict(image.file)

    return PredictionResponse(
        modelId=runtime.model_id,
        category=runtime.category,
        score=prediction.anomaly_score,
        threshold=prediction.threshold,
        isAnomalous=prediction.is_anomalous,
    )