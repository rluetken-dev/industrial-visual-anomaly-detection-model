from collections.abc import Iterable

import torch

from .embedding_extractor import PatchEmbeddingExtractor


def build_feature_memory(
    batches: Iterable[tuple[torch.Tensor, object]],
    embedding_extractor: PatchEmbeddingExtractor,
) -> torch.Tensor:
    """Build a complete normal feature memory from preprocessed image batches."""

    embedding_batches: list[torch.Tensor] = []

    embedding_extractor.eval()

    with torch.inference_mode():
        for image_batch, _ in batches:
            embeddings = embedding_extractor(image_batch)
            embedding_batches.append(embeddings.cpu())

    if not embedding_batches:
        raise ValueError("Feature memory requires at least one image batch.")

    return torch.cat(embedding_batches, dim=0)