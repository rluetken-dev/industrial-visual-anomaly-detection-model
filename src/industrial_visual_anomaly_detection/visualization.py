import torch
import torch.nn.functional as functional


def normalize_anomaly_map(
    anomaly_map: torch.Tensor,
) -> torch.Tensor:
    """Normalize one anomaly map to the range from zero to one."""

    if anomaly_map.ndim != 2:
        raise ValueError("Anomaly map must be two-dimensional.")

    if anomaly_map.numel() == 0:
        raise ValueError("Anomaly map must not be empty.")

    if not torch.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map must contain only finite values.")

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()
    value_range = maximum - minimum

    if value_range.item() == 0.0:
        return torch.zeros_like(anomaly_map)

    return (anomaly_map - minimum) / value_range


def resize_anomaly_map(
    anomaly_map: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Resize one anomaly map using bilinear interpolation."""

    if anomaly_map.ndim != 2:
        raise ValueError("Anomaly map must be two-dimensional.")

    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("Output dimensions must be greater than zero.")

    resized_map = functional.interpolate(
        anomaly_map[None, None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    return resized_map[0, 0]


def colorize_anomaly_map(
    normalized_map: torch.Tensor,
) -> torch.Tensor:
    """Convert a normalized anomaly map into an RGB heatmap."""

    if normalized_map.ndim != 2:
        raise ValueError(
            "Normalized anomaly map must be two-dimensional."
        )

    if normalized_map.numel() == 0:
        raise ValueError("Normalized anomaly map must not be empty.")

    if not torch.isfinite(normalized_map).all():
        raise ValueError(
            "Normalized anomaly map must contain only finite values."
        )

    if (
        normalized_map.min().item() < 0.0
        or normalized_map.max().item() > 1.0
    ):
        raise ValueError(
            "Normalized anomaly map values must be between zero and one."
        )

    red = normalized_map
    blue = 1.0 - normalized_map
    green = 1.0 - torch.abs(
        2.0 * normalized_map - 1.0
    )

    return torch.stack(
        [red, green, blue],
        dim=0,
    )


def create_heatmap_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    opacity: float = 0.5,
) -> torch.Tensor:
    """Blend an RGB image with an equally sized RGB heatmap."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Image must have shape (3, height, width).")

    if heatmap.shape != image.shape:
        raise ValueError(
            "Image and heatmap must have equal shapes."
        )

    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Opacity must be between zero and one.")

    if not torch.isfinite(image).all():
        raise ValueError("Image must contain only finite values.")

    if not torch.isfinite(heatmap).all():
        raise ValueError("Heatmap must contain only finite values.")

    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError(
            "Image values must be between zero and one."
        )

    if heatmap.min().item() < 0.0 or heatmap.max().item() > 1.0:
        raise ValueError(
            "Heatmap values must be between zero and one."
        )

    return (
        (1.0 - opacity) * image
        + opacity * heatmap
    ).clamp(0.0, 1.0)


def normalize_anomaly_map_by_threshold(
    anomaly_map: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Normalize anomaly scores against a fixed decision threshold."""

    if anomaly_map.ndim != 2:
        raise ValueError("Anomaly map must be two-dimensional.")

    if anomaly_map.numel() == 0:
        raise ValueError("Anomaly map must not be empty.")

    if not torch.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map must contain only finite values.")

    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("Threshold must be finite.")

    if threshold <= 0.0:
        raise ValueError("Threshold must be greater than zero.")

    return (anomaly_map / threshold).clamp(0.0, 1.0)