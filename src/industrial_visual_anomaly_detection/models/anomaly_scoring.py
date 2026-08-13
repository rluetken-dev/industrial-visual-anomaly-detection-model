import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from .nearest_neighbors import compute_nearest_neighbor_distances


@dataclass(frozen=True)
class AnomalyScoreBatch:
    """Contain patch-grid and image-level anomaly scores."""

    patch_scores: torch.Tensor
    image_scores: torch.Tensor


def aggregate_top_patch_scores(
    patch_scores: torch.Tensor,
    top_fraction: float,
) -> torch.Tensor:
    """Average the highest-scoring patch fraction for each image."""

    if patch_scores.ndim != 3:
        raise ValueError(
            "Patch scores must have shape (images, height, width)."
        )

    if patch_scores.numel() == 0:
        raise ValueError("Patch scores must not be empty.")

    if not torch.isfinite(patch_scores).all():
        raise ValueError(
            "Patch scores must contain only finite values."
        )

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(
            "Top fraction must be greater than zero and at most one."
        )

    flattened_scores = patch_scores.flatten(start_dim=1)
    patch_count = flattened_scores.shape[1]

    selected_patch_count = max(
        1,
        math.ceil(patch_count * top_fraction),
    )

    top_scores = torch.topk(
        flattened_scores,
        k=selected_patch_count,
        dim=1,
    ).values

    return top_scores.mean(dim=1)


def aggregate_patch_scores(
    patch_scores: torch.Tensor,
    method: str = "maximum",
    top_fraction: float = 0.01,
) -> torch.Tensor:
    """Aggregate patch grids into one anomaly score per image."""

    if method == "maximum":
        if patch_scores.ndim != 3:
            raise ValueError(
                "Patch scores must have shape "
                "(images, height, width)."
            )

        if patch_scores.numel() == 0:
            raise ValueError("Patch scores must not be empty.")

        if not torch.isfinite(patch_scores).all():
            raise ValueError(
                "Patch scores must contain only finite values."
            )

        return patch_scores.flatten(
            start_dim=1
        ).max(dim=1).values

    if method == "top_fraction_mean":
        return aggregate_top_patch_scores(
            patch_scores,
            top_fraction=top_fraction,
        )

    raise ValueError(
        f"Unsupported patch-score aggregation method: {method}"
    )


def compute_anomaly_scores(
    query_embeddings: torch.Tensor,
    feature_memory: torch.Tensor,
    patch_grid_size: tuple[int, int] = (28, 28),
    memory_chunk_size: int = 4096,
    aggregation_method: str = "maximum",
    top_fraction: float = 0.01,
) -> AnomalyScoreBatch:
    """Compute patch-grid and aggregated image-level anomaly scores."""

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

    image_scores = aggregate_patch_scores(
        patch_scores,
        method=aggregation_method,
        top_fraction=top_fraction,
    )

    return AnomalyScoreBatch(
        patch_scores=patch_scores,
        image_scores=image_scores,
    )


def compute_patch_scores_for_batches(
    batches: Iterable[tuple[torch.Tensor, Sequence[str]]],
    embedding_extractor: torch.nn.Module,
    feature_memory: torch.Tensor,
    memory_chunk_size: int = 4096,
    patch_grid_size: tuple[int, int] = (28, 28),
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Compute patch-score grids for ordered image batches."""

    patch_score_batches: list[torch.Tensor] = []
    image_paths: list[str] = []

    embedding_extractor.eval()

    with torch.inference_mode():
        for image_batch, batch_paths in batches:
            query_embeddings = embedding_extractor(image_batch)

            result = compute_anomaly_scores(
                query_embeddings,
                feature_memory,
                patch_grid_size=patch_grid_size,
                memory_chunk_size=memory_chunk_size,
            )

            patch_score_batches.append(result.patch_scores.cpu())
            image_paths.extend(batch_paths)

    if not patch_score_batches:
        raise ValueError(
            "Patch scoring requires at least one batch."
        )

    return torch.cat(patch_score_batches), tuple(image_paths)


def compute_image_scores_for_batches(
    batches: Iterable[tuple[torch.Tensor, Sequence[str]]],
    embedding_extractor: torch.nn.Module,
    feature_memory: torch.Tensor,
    memory_chunk_size: int = 4096,
    patch_grid_size: tuple[int, int] = (28, 28),
    aggregation_method: str = "maximum",
    top_fraction: float = 0.01,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Compute aggregated image-level scores for ordered image batches."""

    patch_scores, image_paths = compute_patch_scores_for_batches(
        batches,
        embedding_extractor,
        feature_memory,
        memory_chunk_size=memory_chunk_size,
        patch_grid_size=patch_grid_size,
    )

    image_scores = aggregate_patch_scores(
        patch_scores,
        method=aggregation_method,
        top_fraction=top_fraction,
    )

    return image_scores, image_paths
