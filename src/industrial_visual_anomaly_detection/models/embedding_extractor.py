import torch

from .feature_extractor import (
    ResNet18FeatureExtractor,
    create_resnet18_feature_extractor,
)
from .patch_embeddings import create_patch_embeddings


class PatchEmbeddingExtractor(torch.nn.Module):
    """Convert preprocessed image tensors into local patch embeddings."""

    def __init__(
        self,
        feature_extractor: ResNet18FeatureExtractor,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        layer2_features, layer3_features = self.feature_extractor(image)

        return create_patch_embeddings(
            layer2_features,
            layer3_features,
        )


def create_resnet18_patch_embedding_extractor() -> PatchEmbeddingExtractor:
    """Create the frozen patch-embedding extractor for the first baseline."""

    feature_extractor = create_resnet18_feature_extractor()
    embedding_extractor = PatchEmbeddingExtractor(feature_extractor)
    embedding_extractor.eval()

    return embedding_extractor