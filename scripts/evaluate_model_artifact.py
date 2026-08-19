import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from industrial_visual_anomaly_detection.artifacts import (
    load_model_artifact,
)
from industrial_visual_anomaly_detection.datasets import (
    ImagePathDataset,
    LabeledImage,
    load_labeled_image_manifest,
)
from industrial_visual_anomaly_detection.evaluation import (
    classify_anomaly_scores,
    compute_binary_classification_metrics,
)
from industrial_visual_anomaly_detection.models import (
    aggregate_top_patch_scores,
    compute_patch_scores_for_batches,
    create_resnet18_patch_embedding_extractor,
)
from industrial_visual_anomaly_detection.preprocessing import (
    create_image_preprocessing,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Evaluate an exported anomaly-detection model artifact "
            "using a dataset-independent labeled-image manifest."
        )
    )
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help=(
            "Directory containing metadata.json and "
            "feature_memory.pt."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help=(
            "Dataset directory against which manifest image "
            "paths are resolved."
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "CSV file containing image, group, and "
            "is_anomalous columns."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--memory-chunk-size",
        type=int,
        default=4096,
    )
    return parser.parse_args()


def select_group_indices(
    labeled_images: tuple[LabeledImage, ...],
    group: str,
) -> list[int]:
    return [
        index
        for index, image in enumerate(labeled_images)
        if image.group == group
    ]


def select_misclassified_indices(
    predictions: torch.Tensor,
    expected_labels: torch.Tensor,
    predicted_anomalous: bool,
    expected_anomalous: bool,
) -> list[int]:
    return [
        index
        for index, (prediction, expected_label) in enumerate(
            zip(predictions, expected_labels, strict=True)
        )
        if (
            bool(prediction.item()) == predicted_anomalous
            and bool(expected_label.item()) == expected_anomalous
        )
    ]


def print_misclassified_images(
    title: str,
    indices: list[int],
    scores: torch.Tensor,
    labeled_images: tuple[LabeledImage, ...],
) -> None:
    print()
    print(f"{title}:")

    if not indices:
        print("- none")
        return

    for index in indices:
        image = labeled_images[index]

        print(
            f"- score={scores[index].item():.6f}, "
            f"group={image.group}, "
            f"path={image.path}"
        )


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    if arguments.memory_chunk_size <= 0:
        raise ValueError(
            "Memory chunk size must be greater than zero."
        )

    labeled_images = load_labeled_image_manifest(
        arguments.dataset_root,
        arguments.manifest,
    )

    if not any(
        not image.is_anomalous
        for image in labeled_images
    ):
        raise ValueError(
            "Evaluation manifest contains no normal images."
        )

    if not any(
        image.is_anomalous
        for image in labeled_images
    ):
        raise ValueError(
            "Evaluation manifest contains no anomalous images."
        )

    artifact_load_start = time.perf_counter()
    artifact = load_model_artifact(arguments.artifact)
    artifact_load_seconds = (
        time.perf_counter() - artifact_load_start
    )

    metadata = artifact.metadata

    if metadata.aggregation_method != "top_fraction_mean":
        raise ValueError(
            "Unsupported artifact aggregation method: "
            f"{metadata.aggregation_method}"
        )

    preprocessing = create_image_preprocessing(
        input_size=(
            metadata.input_size,
            metadata.input_size,
        )
    )

    image_paths = tuple(
        image.path
        for image in labeled_images
    )
    dataset = ImagePathDataset(
        image_paths,
        preprocessing,
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    extractor_start = time.perf_counter()
    embedding_extractor = (
        create_resnet18_patch_embedding_extractor()
    )
    extractor_creation_seconds = (
        time.perf_counter() - extractor_start
    )

    scoring_start = time.perf_counter()
    patch_scores, scored_paths = (
        compute_patch_scores_for_batches(
            loader,
            embedding_extractor,
            artifact.feature_memory,
            memory_chunk_size=(
                arguments.memory_chunk_size
            ),
            patch_grid_size=metadata.patch_grid_size,
        )
    )
    scoring_seconds = time.perf_counter() - scoring_start

    expected_paths = tuple(
        str(path)
        for path in image_paths
    )

    if scored_paths != expected_paths:
        raise RuntimeError(
            "Scored image paths do not match the "
            "manifest order."
        )

    scores = aggregate_top_patch_scores(
        patch_scores,
        top_fraction=metadata.top_fraction,
    )
    predictions = classify_anomaly_scores(
        scores,
        metadata.threshold,
    )
    expected_labels = torch.tensor(
        [
            image.is_anomalous
            for image in labeled_images
        ],
        dtype=torch.bool,
    )

    metrics = compute_binary_classification_metrics(
        predictions,
        expected_labels,
    )

    normal_count = sum(
        not image.is_anomalous
        for image in labeled_images
    )
    anomalous_count = sum(
        image.is_anomalous
        for image in labeled_images
    )

    specificity_denominator = (
        metrics.true_negatives
        + metrics.false_positives
    )
    specificity = (
        metrics.true_negatives
        / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    print(f"Dataset: {metadata.dataset}")
    print(f"Category: {metadata.category}")
    print(f"Artifact: {arguments.artifact.resolve()}")
    print(f"Manifest: {arguments.manifest.resolve()}")
    print(
        f"Input size: "
        f"{metadata.input_size} x {metadata.input_size}"
    )
    print(f"Patch grid size: {metadata.patch_grid_size}")
    print(
        "Aggregation method: "
        f"{metadata.aggregation_method}"
    )
    print(f"Top fraction: {metadata.top_fraction:.4f}")
    print(f"Threshold: {metadata.threshold:.6f}")
    print(
        "Feature memory shape: "
        f"{tuple(artifact.feature_memory.shape)}"
    )
    print(f"Evaluation images: {len(labeled_images)}")
    print(f"Normal images: {normal_count}")
    print(f"Anomalous images: {anomalous_count}")
    print(
        "Artifact load time: "
        f"{artifact_load_seconds:.2f} seconds"
    )
    print(
        "Feature extractor creation time: "
        f"{extractor_creation_seconds:.2f} seconds"
    )
    print(
        f"Evaluation scoring time: "
        f"{scoring_seconds:.2f} seconds"
    )

    print()
    print("Score distribution by group:")

    groups = sorted(
        {
            image.group
            for image in labeled_images
        }
    )

    for group in groups:
        group_indices = select_group_indices(
            labeled_images,
            group,
        )
        group_scores = scores[group_indices]

        print(
            f"- {group}: "
            f"count={group_scores.numel()}, "
            f"min={group_scores.min().item():.6f}, "
            f"mean={group_scores.mean().item():.6f}, "
            f"max={group_scores.max().item():.6f}"
        )

    print()
    print("Classification results:")
    print(
        f"- true positives: "
        f"{metrics.true_positives}"
    )
    print(
        f"- true negatives: "
        f"{metrics.true_negatives}"
    )
    print(
        f"- false positives: "
        f"{metrics.false_positives}"
    )
    print(
        f"- false negatives: "
        f"{metrics.false_negatives}"
    )
    print(f"- accuracy: {metrics.accuracy:.4f}")
    print(f"- precision: {metrics.precision:.4f}")
    print(f"- recall: {metrics.recall:.4f}")
    print(f"- specificity: {specificity:.4f}")
    print(f"- F1 score: {metrics.f1_score:.4f}")

    print()
    print("Predicted anomaly rate by group:")

    for group in groups:
        group_indices = select_group_indices(
            labeled_images,
            group,
        )
        group_predictions = predictions[group_indices]
        predicted_anomalies = int(
            group_predictions.sum().item()
        )

        print(
            f"- {group}: "
            f"predicted anomalous="
            f"{predicted_anomalies}/"
            f"{len(group_indices)}, "
            f"rate="
            f"{predicted_anomalies / len(group_indices):.4f}"
        )

    false_positive_indices = select_misclassified_indices(
        predictions,
        expected_labels,
        predicted_anomalous=True,
        expected_anomalous=False,
    )
    false_negative_indices = select_misclassified_indices(
        predictions,
        expected_labels,
        predicted_anomalous=False,
        expected_anomalous=True,
    )

    print_misclassified_images(
        "False positives",
        false_positive_indices,
        scores,
        labeled_images,
    )
    print_misclassified_images(
        "False negatives",
        false_negative_indices,
        scores,
        labeled_images,
    )


if __name__ == "__main__":
    main()