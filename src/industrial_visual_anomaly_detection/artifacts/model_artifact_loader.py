import json
from pathlib import Path

import torch

from .model_artifact import ModelArtifact
from .model_artifact_metadata import ModelArtifactMetadata


def load_model_artifact(
    artifact_directory: Path,
) -> ModelArtifact:
    """Load model metadata and feature memory from an artifact directory."""

    resolved_artifact_directory = artifact_directory.resolve()

    if not resolved_artifact_directory.is_dir():
        raise NotADirectoryError(
            "Model artifact directory does not exist: "
            f"{resolved_artifact_directory}"
        )

    metadata_path = resolved_artifact_directory / "metadata.json"
    feature_memory_path = (
        resolved_artifact_directory / "feature_memory.pt"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata does not exist: {metadata_path}"
        )

    if not feature_memory_path.is_file():
        raise FileNotFoundError(
            f"Feature memory does not exist: {feature_memory_path}"
        )

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata_data = json.load(metadata_file)

    metadata = ModelArtifactMetadata(
        schema_version=int(metadata_data["schema_version"]),
        dataset=str(metadata_data["dataset"]),
        category=str(metadata_data["category"]),
        backbone=str(metadata_data["backbone"]),
        input_size=int(metadata_data["input_size"]),
        patch_grid_size=tuple(
            int(value)
            for value in metadata_data["patch_grid_size"]
        ),
        embedding_dimension=int(
            metadata_data["embedding_dimension"]
        ),
        aggregation_method=str(
            metadata_data["aggregation_method"]
        ),
        top_fraction=float(metadata_data["top_fraction"]),
        threshold=float(metadata_data["threshold"]),
        memory_fraction=float(metadata_data["memory_fraction"]),
        sampling_seed=int(metadata_data["sampling_seed"]),
        feature_memory_entries=int(
            metadata_data["feature_memory_entries"]
        ),
        threshold_method=str(
            metadata_data.get(
                "threshold_method",
                "maximum_normal",
            )
        ),
        threshold_quantile=float(
            metadata_data.get(
                "threshold_quantile",
                1.0,
            )
        ),
    )

    feature_memory = torch.load(
        feature_memory_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(feature_memory, torch.Tensor):
        raise TypeError("Feature-memory file must contain a tensor.")

    if feature_memory.ndim != 2:
        raise ValueError("Feature memory must be two-dimensional.")

    if feature_memory.shape[0] != metadata.feature_memory_entries:
        raise ValueError(
            "Feature-memory entry count does not match the metadata."
        )

    if feature_memory.shape[1] != metadata.embedding_dimension:
        raise ValueError(
            "Feature-memory dimension does not match the metadata."
        )

    if not torch.isfinite(feature_memory).all():
        raise ValueError(
            "Feature memory must contain only finite values."
        )

    return ModelArtifact(
        metadata=metadata,
        feature_memory=feature_memory,
    )