import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceServiceSettings:
    """Contain validated inference-service configuration."""

    artifact_path: Path | None
    memory_chunk_size: int
    model_registry_path: Path | None = None

    def __post_init__(self) -> None:
        configured_source_count = sum(
            path is not None
            for path in (
                self.artifact_path,
                self.model_registry_path,
            )
        )

        if configured_source_count != 1:
            raise ValueError(
                "Exactly one model configuration source is required."
            )

        if self.memory_chunk_size <= 0:
            raise ValueError(
                "Memory chunk size must be greater than zero."
            )

    @classmethod
    def from_environment(cls) -> "InferenceServiceSettings":
        artifact_value = os.environ.get(
            "IVAD_MODEL_ARTIFACT",
            "",
        ).strip()
        registry_value = os.environ.get(
            "IVAD_MODEL_REGISTRY",
            "",
        ).strip()

        if artifact_value and registry_value:
            raise RuntimeError(
                "IVAD_MODEL_ARTIFACT and IVAD_MODEL_REGISTRY "
                "must not both be configured."
            )

        if not artifact_value and not registry_value:
            raise RuntimeError(
                "Exactly one of IVAD_MODEL_ARTIFACT or "
                "IVAD_MODEL_REGISTRY must be configured."
            )

        chunk_size_value = os.environ.get(
            "IVAD_MEMORY_CHUNK_SIZE",
            "4096",
        )

        try:
            memory_chunk_size = int(chunk_size_value)
        except ValueError as exception:
            raise RuntimeError(
                "IVAD_MEMORY_CHUNK_SIZE must be an integer."
            ) from exception

        if memory_chunk_size <= 0:
            raise RuntimeError(
                "IVAD_MEMORY_CHUNK_SIZE must be greater than zero."
            )

        artifact_path = (
            Path(artifact_value).expanduser().resolve()
            if artifact_value
            else None
        )
        model_registry_path = (
            Path(registry_value).expanduser().resolve()
            if registry_value
            else None
        )

        return cls(
            artifact_path=artifact_path,
            memory_chunk_size=memory_chunk_size,
            model_registry_path=model_registry_path,
        )