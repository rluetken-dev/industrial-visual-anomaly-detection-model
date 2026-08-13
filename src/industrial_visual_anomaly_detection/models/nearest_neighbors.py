import torch


def compute_nearest_neighbor_distances(
    query_embeddings: torch.Tensor,
    feature_memory: torch.Tensor,
    memory_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute the exact nearest normal-memory distance for each query embedding."""

    if query_embeddings.ndim != 2:
        raise ValueError("Query embeddings must be a two-dimensional tensor.")

    if feature_memory.ndim != 2:
        raise ValueError("Feature memory must be a two-dimensional tensor.")

    if query_embeddings.shape[1] != feature_memory.shape[1]:
        raise ValueError(
            "Query embeddings and feature memory must use the same feature dimension."
        )

    if query_embeddings.shape[0] == 0:
        raise ValueError("At least one query embedding is required.")

    if feature_memory.shape[0] == 0:
        raise ValueError("Feature memory must contain at least one embedding.")

    if memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    if query_embeddings.device != feature_memory.device:
        raise ValueError(
            "Query embeddings and feature memory must use the same device."
        )

    nearest_distances = torch.full(
        (query_embeddings.shape[0],),
        float("inf"),
        dtype=query_embeddings.dtype,
        device=query_embeddings.device,
    )

    for start_index in range(0, feature_memory.shape[0], memory_chunk_size):
        memory_chunk = feature_memory[
            start_index : start_index + memory_chunk_size
        ]

        distances = torch.cdist(
            query_embeddings,
            memory_chunk,
            p=2,
        )

        chunk_nearest_distances = distances.min(dim=1).values

        nearest_distances = torch.minimum(
            nearest_distances,
            chunk_nearest_distances,
        )

    return nearest_distances