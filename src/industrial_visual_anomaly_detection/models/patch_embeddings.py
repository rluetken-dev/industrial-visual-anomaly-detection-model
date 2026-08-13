import torch
import torch.nn.functional as functional


def create_patch_embeddings(
    layer2_features: torch.Tensor,
    layer3_features: torch.Tensor,
) -> torch.Tensor:
    """Combine multi-scale feature maps into local patch embeddings."""

    resized_layer3_features = functional.interpolate(
        layer3_features,
        size=layer2_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    combined_features = torch.cat(
        [layer2_features, resized_layer3_features],
        dim=1,
    )

    patch_embeddings = (
        combined_features
        .permute(0, 2, 3, 1)
        .reshape(-1, combined_features.shape[1])
    )

    return patch_embeddings