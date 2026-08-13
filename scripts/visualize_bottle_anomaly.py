import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (
    InterpolationMode,
    Resize,
    ToTensor,
)
from torchvision.utils import save_image

from industrial_visual_anomaly_detection.datasets import (
    ImagePathDataset,
    load_split_manifest,
    resolve_dataset_image_paths,
    validate_split_manifest,
)
from industrial_visual_anomaly_detection.models import (
    build_feature_memory,
    compute_anomaly_scores,
    create_resnet18_patch_embedding_extractor,
)
from industrial_visual_anomaly_detection.preprocessing import (
    BOTTLE_INPUT_SIZE,
    create_bottle_preprocessing,
)
from industrial_visual_anomaly_detection.visualization import (
    colorize_anomaly_map,
    create_heatmap_overlay,
    normalize_anomaly_map,
    normalize_anomaly_map_by_threshold,
    resize_anomaly_map,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Create an anomaly heatmap for one Bottle image."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Optional fixed score threshold used as the maximum "
            "of the heatmap color scale."
        ),
    )
    return parser.parse_args()


def load_visualization_image(image_path: Path) -> torch.Tensor:
    """Load one RGB image without ImageNet normalization."""

    resize = Resize(
        size=BOTTLE_INPUT_SIZE,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    to_tensor = ToTensor()

    with Image.open(image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        resized_image = resize(rgb_image)

    return to_tensor(resized_image)


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if arguments.memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    if not 0.0 <= arguments.opacity <= 1.0:
        raise ValueError("Opacity must be between zero and one.")

    if (
        arguments.threshold is not None
        and arguments.threshold <= 0.0
    ):
        raise ValueError("Threshold must be greater than zero.")

    dataset_root = arguments.dataset_root.resolve()
    image_path = arguments.image.resolve()
    output_dir = arguments.output_dir.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    manifest = load_split_manifest(arguments.manifest)
    validate_split_manifest(manifest)

    fitting_paths = resolve_dataset_image_paths(
        dataset_root,
        manifest.fitting_images,
    )

    preprocessing = create_bottle_preprocessing()

    fitting_dataset = ImagePathDataset(
        fitting_paths,
        preprocessing,
    )
    fitting_loader = DataLoader(
        fitting_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    embedding_extractor = (
        create_resnet18_patch_embedding_extractor()
    )

    memory_start = time.perf_counter()

    feature_memory = build_feature_memory(
        fitting_loader,
        embedding_extractor,
    )

    memory_seconds = time.perf_counter() - memory_start

    with Image.open(image_path) as source_image:
        model_image = preprocessing(
            source_image.convert("RGB")
        ).unsqueeze(0)

    with torch.inference_mode():
        query_embeddings = embedding_extractor(model_image)

        result = compute_anomaly_scores(
            query_embeddings,
            feature_memory,
            memory_chunk_size=arguments.memory_chunk_size,
        )

    patch_scores = result.patch_scores[0]

    resized_scores = resize_anomaly_map(
        patch_scores,
        output_size=BOTTLE_INPUT_SIZE,
    )

    if arguments.threshold is None:
        normalized_scores = normalize_anomaly_map(
            resized_scores
        )
        color_scale = "relative per-image minimum and maximum"
    else:
        normalized_scores = normalize_anomaly_map_by_threshold(
            resized_scores,
            arguments.threshold,
        )
        color_scale = f"fixed threshold {arguments.threshold:.6f}"

    heatmap = colorize_anomaly_map(normalized_scores)

    visualization_image = load_visualization_image(image_path)

    overlay = create_heatmap_overlay(
        visualization_image,
        heatmap,
        opacity=arguments.opacity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    original_path = output_dir / f"{stem}_original.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    save_image(visualization_image, original_path)
    save_image(heatmap, heatmap_path)
    save_image(overlay, overlay_path)

    print(f"Image: {image_path}")
    print(f"Feature memory shape: {tuple(feature_memory.shape)}")
    print(f"Feature memory build time: {memory_seconds:.2f} seconds")
    print(f"Patch score shape: {tuple(patch_scores.shape)}")
    print(f"Minimum patch score: {patch_scores.min().item():.6f}")
    print(f"Maximum patch score: {patch_scores.max().item():.6f}")
    print(f"Image anomaly score: {result.image_scores[0].item():.6f}")
    print(f"Heatmap color scale: {color_scale}")
    print(f"Original image: {original_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
