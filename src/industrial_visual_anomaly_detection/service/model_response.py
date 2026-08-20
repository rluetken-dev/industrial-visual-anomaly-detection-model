from pydantic import BaseModel


class ModelResponse(BaseModel):
    """Represent one model available for inference."""

    id: str
    displayName: str
    category: str
    inputSize: int
    isDefault: bool


class ModelCatalogResponse(BaseModel):
    """Represent the available inference-model catalog."""

    defaultModelId: str
    models: list[ModelResponse]