from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from .nearest_neighbors import compute_nearest_neighbor_distances


@dataclass(frozen=True)
class AnomalyScoreBatch:
    """Contain patch-grid and image-level anomaly scores."""

    patch_scores: torch.Tensor
    image_scores: torch.Tensor


def compute_anomaly_scores(
    query_embeddings: torch.Tensor,
    feature_memory: torch.Tensor,
    patch_grid_size: tuple[int, int] = (28, 28),
    memory_chunk_size: int = 4096,
) -> AnomalyScoreBatch:
    """Compute patch-grid scores and maximum image-level anomaly scores."""

    patches_per_image = patch_grid_size[0] * patch_grid_size[1]

    if patches_per_image <= 0:
        raise ValueError("Patch grid dimensions must be greater than zero.")

    if query_embeddings.shape[0] % patches_per_image != 0:
        raise ValueError(
            "Query embedding count must be divisible by the patch-grid size."
        )

    nearest_distances = compute_nearest_neighbor_distances(
        query_embeddings,
        feature_memory,
        memory_chunk_size=memory_chunk_size,
    )

    patch_scores = nearest_distances.reshape(
        -1,
        patch_grid_size[0],
        patch_grid_size[1],
    )

    image_scores = patch_scores.flatten(start_dim=1).max(dim=1).values

    return AnomalyScoreBatch(
        patch_scores=patch_scores,
        image_scores=image_scores,
    )


def compute_image_scores_for_batches(
    batches: Iterable[tuple[torch.Tensor, Sequence[str]]],
    embedding_extractor: torch.nn.Module,
    feature_memory: torch.Tensor,
    memory_chunk_size: int = 4096,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Compute image-level anomaly scores for ordered image batches."""

    score_batches: list[torch.Tensor] = []
    image_paths: list[str] = []

    embedding_extractor.eval()

    with torch.inference_mode():
        for image_batch, batch_paths in batches:
            query_embeddings = embedding_extractor(image_batch)

            result = compute_anomaly_scores(
                query_embeddings,
                feature_memory,
                memory_chunk_size=memory_chunk_size,
            )

            score_batches.append(result.image_scores.cpu())
            image_paths.extend(batch_paths)

    if not score_batches:
        raise ValueError("Image scoring requires at least one batch.")

    return torch.cat(score_batches), tuple(image_paths)