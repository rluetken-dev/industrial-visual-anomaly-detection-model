from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .artifacts import ModelArtifact
from .models import (
    aggregate_top_patch_scores,
    compute_anomaly_scores,
)
from .preprocessing import create_image_preprocessing


@dataclass(frozen=True)
class AnomalyPrediction:
    """Contain the anomaly-detection result for one image."""

    image_path: Path
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    patch_scores: torch.Tensor


def predict_image(
    artifact: ModelArtifact,
    image_path: Path,
    embedding_extractor: torch.nn.Module,
    memory_chunk_size: int = 4096,
) -> AnomalyPrediction:
    """Predict whether one image contains a visual anomaly."""

    resolved_image_path = image_path.resolve()

    if not resolved_image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {resolved_image_path}"
        )

    if memory_chunk_size <= 0:
        raise ValueError(
            "Memory chunk size must be greater than zero."
        )

    metadata = artifact.metadata

    if metadata.aggregation_method != "top_fraction_mean":
        raise ValueError(
            "Unsupported aggregation method: "
            f"{metadata.aggregation_method}"
        )

    preprocessing = create_image_preprocessing(
        input_size=(
            metadata.input_size,
            metadata.input_size,
        )
    )

    with Image.open(resolved_image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        image_tensor = preprocessing(rgb_image).unsqueeze(0)

    embedding_extractor.eval()

    with torch.inference_mode():
        query_embeddings = embedding_extractor(image_tensor)

        score_batch = compute_anomaly_scores(
            query_embeddings,
            artifact.feature_memory,
            patch_grid_size=metadata.patch_grid_size,
            memory_chunk_size=memory_chunk_size,
        )

        image_scores = aggregate_top_patch_scores(
            score_batch.patch_scores,
            top_fraction=metadata.top_fraction,
        )

    anomaly_score = float(image_scores[0].item())
    is_anomalous = anomaly_score > metadata.threshold

    return AnomalyPrediction(
        image_path=resolved_image_path,
        anomaly_score=anomaly_score,
        threshold=metadata.threshold,
        is_anomalous=is_anomalous,
        patch_scores=score_batch.patch_scores[0].cpu(),
    )