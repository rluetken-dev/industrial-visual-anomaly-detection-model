from dataclasses import dataclass

from industrial_visual_anomaly_detection.artifacts import (
    load_model_artifact,
)
from industrial_visual_anomaly_detection.models import (
    create_resnet18_patch_embedding_extractor,
)

from .model_registry_config import ModelRegistryConfiguration
from .runtime import InferenceRuntime


@dataclass(frozen=True)
class AvailableModel:
    """Describe one loaded model exposed by the service."""

    model_id: str
    display_name: str
    category: str
    input_size: int
    is_default: bool


class UnknownModelError(LookupError):
    """Indicate that a requested model is not registered."""


class InferenceRuntimeRegistry:
    """Hold loaded inference runtimes indexed by stable model ID."""

    def __init__(
        self,
        configuration: ModelRegistryConfiguration,
        runtimes: dict[str, InferenceRuntime],
    ) -> None:
        enabled_model_ids = {
            model.model_id
            for model in configuration.enabled_models
        }

        if set(runtimes) != enabled_model_ids:
            raise ValueError(
                "Loaded runtime IDs must match enabled registry models."
            )

        self._configuration = configuration
        self._runtimes = dict(runtimes)

    @classmethod
    def load(
        cls,
        configuration: ModelRegistryConfiguration,
        memory_chunk_size: int,
    ) -> "InferenceRuntimeRegistry":
        """Load every enabled registry artifact."""

        if memory_chunk_size <= 0:
            raise ValueError(
                "Memory chunk size must be greater than zero."
            )

        runtimes: dict[str, InferenceRuntime] = {}

        for model in configuration.enabled_models:
            artifact = load_model_artifact(
                model.artifact_path
            )
            embedding_extractor = (
                create_resnet18_patch_embedding_extractor()
            )

            runtimes[model.model_id] = InferenceRuntime(
                artifact_path=model.artifact_path,
                artifact=artifact,
                embedding_extractor=embedding_extractor,
                memory_chunk_size=memory_chunk_size,
                model_id=model.model_id,
            )

        return cls(
            configuration=configuration,
            runtimes=runtimes,
        )

    @property
    def default_model_id(self) -> str:
        """Return the configured default model ID."""

        return self._configuration.default_model_id

    @property
    def available_models(self) -> tuple[AvailableModel, ...]:
        """Return descriptions of all enabled loaded models."""

        return tuple(
            AvailableModel(
                model_id=model.model_id,
                display_name=model.display_name,
                category=self._runtimes[
                    model.model_id
                ].category,
                input_size=self._runtimes[
                    model.model_id
                ].input_size,
                is_default=(
                    model.model_id
                    == self.default_model_id
                ),
            )
            for model in self._configuration.enabled_models
        )

    def get_runtime(
        self,
        model_id: str | None = None,
    ) -> InferenceRuntime:
        """Return the requested runtime or the configured default."""

        selected_model_id = (
            self.default_model_id
            if model_id is None
            else model_id.strip()
        )

        if not selected_model_id:
            raise UnknownModelError(
                "Requested model ID must not be empty."
            )

        try:
            return self._runtimes[selected_model_id]
        except KeyError as exception:
            raise UnknownModelError(
                f"Unknown model ID: {selected_model_id}"
            ) from exception