import torch


def sample_feature_memory(
    feature_memory: torch.Tensor,
    fraction: float,
    seed: int = 42,
) -> torch.Tensor:
    """Select a deterministic random subset of feature-memory entries."""

    if feature_memory.ndim != 2:
        raise ValueError(
            "Feature memory must have shape (entries, features)."
        )

    if feature_memory.shape[0] == 0:
        raise ValueError("Feature memory must not be empty.")

    if not torch.isfinite(feature_memory).all():
        raise ValueError(
            "Feature memory must contain only finite values."
        )

    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "Sampling fraction must be greater than zero "
            "and at most one."
        )

    if fraction == 1.0:
        return feature_memory

    sampled_entry_count = max(
        1,
        round(feature_memory.shape[0] * fraction),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    sampled_indices = torch.randperm(
        feature_memory.shape[0],
        generator=generator,
    )[:sampled_entry_count]

    return feature_memory[sampled_indices]