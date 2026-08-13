from dataclasses import dataclass

import torch

from .model_artifact_metadata import ModelArtifactMetadata


@dataclass(frozen=True)
class ModelArtifact:
    """Contain metadata and feature memory for an exported model."""

    metadata: ModelArtifactMetadata
    feature_memory: torch.Tensor