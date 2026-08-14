import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceServiceSettings:
    """Contain validated inference-service configuration."""

    artifact_path: Path
    memory_chunk_size: int

    @classmethod
    def from_environment(cls) -> "InferenceServiceSettings":
        artifact_value = os.environ.get("IVAD_MODEL_ARTIFACT", "").strip()

        if not artifact_value:
            raise RuntimeError(
                "IVAD_MODEL_ARTIFACT must contain a model artifact path."
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

        return cls(
            artifact_path=Path(artifact_value).expanduser().resolve(),
            memory_chunk_size=memory_chunk_size,
        )