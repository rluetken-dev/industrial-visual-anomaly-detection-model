import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from industrial_visual_anomaly_detection.datasets import (
    ImagePathDataset,
    discover_mvtec_ad_test_images,
    load_split_manifest,
    resolve_dataset_image_paths,
    validate_split_manifest,
)
from industrial_visual_anomaly_detection.models import (
    build_feature_memory,
    compute_image_scores_for_batches,
    create_resnet18_patch_embedding_extractor,
)
from industrial_visual_anomaly_detection.preprocessing import (
    create_bottle_preprocessing,
)
from industrial_visual_anomaly_detection.evaluation import (
    classify_anomaly_scores,
    compute_binary_classification_metrics,
    select_maximum_normal_threshold,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Build the Bottle feature memory and evaluate the normal "
            "validation partition and labeled test partition."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if arguments.memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    manifest = load_split_manifest(arguments.manifest)
    validate_split_manifest(manifest)

    dataset_root = arguments.dataset_root.resolve()

    fitting_paths = resolve_dataset_image_paths(
        dataset_root,
        manifest.fitting_images,
    )
    validation_paths = resolve_dataset_image_paths(
        dataset_root,
        manifest.validation_images,
    )

    labeled_test_images = discover_mvtec_ad_test_images(
        dataset_root,
        manifest.category,
    )
    test_paths = tuple(
        image.path for image in labeled_test_images
    )

    preprocessing = create_bottle_preprocessing()

    fitting_dataset = ImagePathDataset(
        fitting_paths,
        preprocessing,
    )
    validation_dataset = ImagePathDataset(
        validation_paths,
        preprocessing,
    )
    test_dataset = ImagePathDataset(
        test_paths,
        preprocessing,
    )

    fitting_loader = DataLoader(
        fitting_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
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

    validation_scoring_start = time.perf_counter()

    validation_scores, validation_scored_paths = (
        compute_image_scores_for_batches(
            validation_loader,
            embedding_extractor,
            feature_memory,
            memory_chunk_size=arguments.memory_chunk_size,
        )
    )

    validation_scoring_seconds = (
        time.perf_counter() - validation_scoring_start
    )

    expected_validation_paths = tuple(
        str(path) for path in validation_paths
    )

    if validation_scored_paths != expected_validation_paths:
        raise RuntimeError(
            "Validation score paths do not match the manifest order."
        )

    test_scoring_start = time.perf_counter()

    test_scores, test_scored_paths = compute_image_scores_for_batches(
        test_loader,
        embedding_extractor,
        feature_memory,
        memory_chunk_size=arguments.memory_chunk_size,
    )

    test_scoring_seconds = time.perf_counter() - test_scoring_start

    expected_test_paths = tuple(
        str(path) for path in test_paths
    )

    if test_scored_paths != expected_test_paths:
        raise RuntimeError(
            "Test score paths do not match the discovered test-image order."
        )

    threshold = select_maximum_normal_threshold(
        validation_scores
    )

    test_predictions = classify_anomaly_scores(
        test_scores,
        threshold,
    )

    expected_test_labels = torch.tensor(
        [
            image.is_anomalous
            for image in labeled_test_images
        ],
        dtype=torch.bool,
    )

    metrics = compute_binary_classification_metrics(
        test_predictions,
        expected_test_labels,
    )

    sorted_validation_scores, sorted_validation_indices = torch.sort(
        validation_scores,
        descending=True,
    )

    print(f"Dataset: {manifest.dataset}")
    print(f"Category: {manifest.category}")
    print(f"Fitting images: {len(fitting_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Feature memory shape: {tuple(feature_memory.shape)}")
    print(
        "Feature memory build time: "
        f"{memory_seconds:.2f} seconds"
    )
    print(
        "Validation scoring time: "
        f"{validation_scoring_seconds:.2f} seconds"
    )
    print()
    print("Normal validation score distribution:")
    print(f"Minimum: {validation_scores.min().item():.6f}")
    print(f"Mean: {validation_scores.mean().item():.6f}")
    print(f"Median: {validation_scores.median().item():.6f}")
    print(
        "Standard deviation: "
        f"{validation_scores.std(unbiased=False).item():.6f}"
    )
    print(
        "95th percentile: "
        f"{torch.quantile(validation_scores, 0.95).item():.6f}"
    )
    print(f"Maximum: {validation_scores.max().item():.6f}")
    print()
    print("Five highest normal validation scores:")

    for score, index in zip(
        sorted_validation_scores[:5],
        sorted_validation_indices[:5],
        strict=True,
    ):
        print(
            f"- {score.item():.6f}: "
            f"{validation_scored_paths[index.item()]}"
        )

    print()
    print(f"Test images: {len(test_dataset)}")
    print(f"Test scoring time: {test_scoring_seconds:.2f} seconds")
    print()
    print("Test score distribution by group:")

    groups = sorted(
        {image.group for image in labeled_test_images}
    )

    for group in groups:
        group_indices = [
            index
            for index, image in enumerate(labeled_test_images)
            if image.group == group
        ]

        group_scores = test_scores[group_indices]

        print(
            f"- {group}: "
            f"count={group_scores.numel()}, "
            f"min={group_scores.min().item():.6f}, "
            f"mean={group_scores.mean().item():.6f}, "
            f"max={group_scores.max().item():.6f}"
        )

    print()
    print("Threshold and classification results:")
    print(f"Threshold: {threshold:.6f}")
    print(f"True positives: {metrics.true_positives}")
    print(f"True negatives: {metrics.true_negatives}")
    print(f"False positives: {metrics.false_positives}")
    print(f"False negatives: {metrics.false_negatives}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1 score: {metrics.f1_score:.4f}")
    print()
    print("Detection rate by test group:")

    for group in groups:
        group_indices = [
            index
            for index, image in enumerate(labeled_test_images)
            if image.group == group
        ]

        group_predictions = test_predictions[group_indices]
        predicted_anomalies = int(
            group_predictions.sum().item()
        )

        print(
            f"- {group}: "
            f"predicted anomalous={predicted_anomalies}/"
            f"{len(group_indices)}, "
            f"rate={predicted_anomalies / len(group_indices):.4f}"
        )

    false_negative_indices = [
        index
        for index, (
            prediction,
            expected_label,
        ) in enumerate(
            zip(
                test_predictions,
                expected_test_labels,
                strict=True,
            )
        )
        if not prediction.item() and expected_label.item()
    ]

    print()
    print("False negatives:")

    if not false_negative_indices:
        print("- none")
    else:
        for index in false_negative_indices:
            print(
                f"- score={test_scores[index].item():.6f}, "
                f"group={labeled_test_images[index].group}, "
                f"path={test_scored_paths[index]}"
            )


if __name__ == "__main__":
    main()