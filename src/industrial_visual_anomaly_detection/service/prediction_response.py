from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Represent one inference response for the ASP.NET Core backend."""

    modelId: str
    category: str
    score: float
    threshold: float
    isAnomalous: bool