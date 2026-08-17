from pydantic import BaseModel


class HeatmapResponse(BaseModel):
    """Represent one encoded anomaly heatmap."""

    contentType: str
    width: int
    height: int
    dataBase64: str


class PredictionResponse(BaseModel):
    """Represent one inference response for the ASP.NET Core backend."""

    modelId: str
    category: str
    score: float
    threshold: float
    isAnomalous: bool
    heatmap: HeatmapResponse