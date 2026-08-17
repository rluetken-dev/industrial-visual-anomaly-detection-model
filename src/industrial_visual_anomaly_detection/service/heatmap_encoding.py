from base64 import b64encode
from io import BytesIO

import torch
from PIL import Image

from industrial_visual_anomaly_detection.visualization import (
    colorize_anomaly_map,
    normalize_anomaly_map_by_threshold,
    resize_anomaly_map,
)


def encode_heatmap_png_base64(
    patch_scores: torch.Tensor,
    threshold: float,
    output_size: tuple[int, int],
) -> str:
    """Convert patch anomaly scores into a Base64-encoded RGB PNG."""

    normalized_map = normalize_anomaly_map_by_threshold(
        patch_scores,
        threshold=threshold,
    )

    resized_map = resize_anomaly_map(
        normalized_map,
        output_size=output_size,
    )

    heatmap = colorize_anomaly_map(resized_map)

    image_data = (
        heatmap.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )

    with BytesIO() as output:
        Image.fromarray(image_data).save(output, format="PNG")
        return b64encode(output.getvalue()).decode("ascii")