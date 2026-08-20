from pathlib import Path
from threading import Lock
from typing import BinaryIO

from industrial_visual_anomaly_detection.artifacts import (
    ModelArtifact,
    load_model_artifact,
)
from industrial_visual_anomaly_detection.inference import (
    AnomalyPredictionResult,
    predict_image_stream,
)
from industrial_visual_anomaly_detection.models import (
    PatchEmbeddingExtractor,
    create_resnet18_patch_embedding_extractor,
)

from .settings import InferenceServiceSettings


class InferenceRuntime:
    """Hold one loaded model artifact and its reusable feature extractor."""

    def __init__(
        self,
        artifact_path: Path,
        artifact: ModelArtifact,
        embedding_extractor: PatchEmbeddingExtractor,
        memory_chunk_size: int,
        model_id: str | None = None,
    ) -> None:
        self.artifact_path = artifact_path
        self.artifact = artifact
        self.embedding_extractor = embedding_extractor
        self.memory_chunk_size = memory_chunk_size
        resolved_model_id = (
            model_id
            if model_id is not None
            else artifact_path.name
        )

        if not resolved_model_id.strip():
            raise ValueError("Model ID must not be empty.")

        self._model_id = resolved_model_id
        self._prediction_lock = Lock()

    @classmethod
    def load(
        cls,
        settings: InferenceServiceSettings,
    ) -> "InferenceRuntime":
        """Load the configured artifact and create its feature extractor."""

        if settings.artifact_path is None:
            raise ValueError(
                "Single-model runtime requires an artifact path."
            )

        artifact = load_model_artifact(settings.artifact_path)
        embedding_extractor = (
            create_resnet18_patch_embedding_extractor()
        )

        return cls(
            artifact_path=settings.artifact_path,
            artifact=artifact,
            embedding_extractor=embedding_extractor,
            memory_chunk_size=settings.memory_chunk_size,
        )

    @property
    def model_id(self) -> str:
        """Return the artifact directory name as the model identifier."""

        return self._model_id

    @property
    def category(self) -> str:
        """Return the configured model category."""

        return self.artifact.metadata.category

    @property
    def input_size(self) -> int:
        """Return the configured square model input size."""

        return self.artifact.metadata.input_size

    def predict(
        self,
        image_stream: BinaryIO,
    ) -> AnomalyPredictionResult:
        """Run one thread-safe prediction using the loaded model."""

        with self._prediction_lock:
            return predict_image_stream(
                artifact=self.artifact,
                image_stream=image_stream,
                embedding_extractor=self.embedding_extractor,
                memory_chunk_size=self.memory_chunk_size,
            )