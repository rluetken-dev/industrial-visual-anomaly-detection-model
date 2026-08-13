import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model_artifact import ModelArtifact


def save_model_artifact(
    artifact: ModelArtifact,
    output_directory: Path,
) -> Path:
    """Save model metadata and feature memory to one artifact directory."""

    resolved_output_directory = output_directory.resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    feature_memory = artifact.feature_memory

    if feature_memory.ndim != 2:
        raise ValueError("Feature memory must be two-dimensional.")

    if feature_memory.shape[0] == 0:
        raise ValueError("Feature memory must not be empty.")

    if not torch.isfinite(feature_memory).all():
        raise ValueError("Feature memory must contain only finite values.")

    metadata = artifact.metadata

    if metadata.feature_memory_entries != feature_memory.shape[0]:
        raise ValueError(
            "Feature-memory entry count does not match the metadata."
        )

    if metadata.embedding_dimension != feature_memory.shape[1]:
        raise ValueError(
            "Feature-memory dimension does not match the metadata."
        )

    metadata_path = resolved_output_directory / "metadata.json"
    feature_memory_path = (
        resolved_output_directory / "feature_memory.pt"
    )

    metadata_data = asdict(metadata)
    metadata_data["patch_grid_size"] = list(
        metadata.patch_grid_size
    )

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            metadata_data,
            metadata_file,
            indent=2,
        )
        metadata_file.write("\n")

    torch.save(
        feature_memory.detach().cpu(),
        feature_memory_path,
    )

    return resolved_output_directory