import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from industrial_visual_anomaly_detection.datasets import (
    ImagePathDataset,
    LabeledImage,
    discover_mvtec_ad_test_images,
    load_split_manifest,
    resolve_dataset_image_paths,
    validate_split_manifest,
)
from industrial_visual_anomaly_detection.evaluation import (
    classify_anomaly_scores,
    compute_binary_classification_metrics,
    select_maximum_normal_threshold,
)
from industrial_visual_anomaly_detection.models import (
    aggregate_top_patch_scores,
    build_feature_memory,
    compute_patch_scores_for_batches,
    create_resnet18_patch_embedding_extractor,
    sample_feature_memory,
)
from industrial_visual_anomaly_detection.preprocessing import (
    create_image_preprocessing,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Compare maximum and top-patch mean aggregation "
            "for one MVTec AD category."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=42)
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


def print_score_rule_results(
    name: str,
    validation_scores: torch.Tensor,
    test_scores: torch.Tensor,
    test_paths: tuple[str, ...],
    labeled_test_images: tuple[LabeledImage, ...],
    expected_test_labels: torch.Tensor,
) -> None:
    threshold = select_maximum_normal_threshold(validation_scores)
    predictions = classify_anomaly_scores(test_scores, threshold)
    metrics = compute_binary_classification_metrics(
        predictions,
        expected_test_labels,
    )

    groups = sorted({image.group for image in labeled_test_images})

    print()
    print(f"=== Aggregation: {name} ===")
    print("Normal validation score distribution:")
    print(f"- minimum: {validation_scores.min().item():.6f}")
    print(f"- mean: {validation_scores.mean().item():.6f}")
    print(f"- median: {validation_scores.median().item():.6f}")
    print(
        "- standard deviation: "
        f"{validation_scores.std(unbiased=False).item():.6f}"
    )
    print(
        "- 95th percentile: "
        f"{torch.quantile(validation_scores, 0.95).item():.6f}"
    )
    print(f"- maximum and threshold: {threshold:.6f}")
    print()
    print("Test score distribution by group:")

    for group in groups:
        group_indices = select_group_indices(labeled_test_images, group)
        group_scores = test_scores[group_indices]

        print(
            f"- {group}: "
            f"count={group_scores.numel()}, "
            f"min={group_scores.min().item():.6f}, "
            f"mean={group_scores.mean().item():.6f}, "
            f"max={group_scores.max().item():.6f}"
        )

    print()
    print("Classification results:")
    print(f"- true positives: {metrics.true_positives}")
    print(f"- true negatives: {metrics.true_negatives}")
    print(f"- false positives: {metrics.false_positives}")
    print(f"- false negatives: {metrics.false_negatives}")
    print(f"- accuracy: {metrics.accuracy:.4f}")
    print(f"- precision: {metrics.precision:.4f}")
    print(f"- recall: {metrics.recall:.4f}")
    print(f"- F1 score: {metrics.f1_score:.4f}")
    print()
    print("Detection rate by test group:")

    for group in groups:
        group_indices = select_group_indices(labeled_test_images, group)
        group_predictions = predictions[group_indices]
        predicted_anomalies = int(group_predictions.sum().item())

        print(
            f"- {group}: "
            f"predicted anomalous={predicted_anomalies}/"
            f"{len(group_indices)}, "
            f"rate={predicted_anomalies / len(group_indices):.4f}"
        )

    false_negative_indices = [
        index
        for index, (prediction, expected_label) in enumerate(
            zip(predictions, expected_test_labels, strict=True)
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
                f"path={test_paths[index]}"
            )


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if arguments.memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    if arguments.input_size <= 0:
        raise ValueError("Input size must be greater than zero.")

    if arguments.input_size % 32 != 0:
        raise ValueError("Input size must be divisible by 32.")

    if not 0.0 < arguments.memory_fraction <= 1.0:
        raise ValueError(
            "Memory fraction must be greater than zero and at most one."
        )

    patch_grid_side = arguments.input_size // 8
    patch_grid_size = (patch_grid_side, patch_grid_side)

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
    test_image_paths = tuple(image.path for image in labeled_test_images)

    preprocessing = create_image_preprocessing(
        input_size=(arguments.input_size, arguments.input_size)
    )

    fitting_dataset = ImagePathDataset(fitting_paths, preprocessing)
    validation_dataset = ImagePathDataset(validation_paths, preprocessing)
    test_dataset = ImagePathDataset(test_image_paths, preprocessing)

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

    embedding_extractor = create_resnet18_patch_embedding_extractor()

    memory_start = time.perf_counter()
    complete_feature_memory = build_feature_memory(
        fitting_loader,
        embedding_extractor,
    )
    complete_feature_memory_shape = tuple(complete_feature_memory.shape)

    feature_memory = sample_feature_memory(
        complete_feature_memory,
        fraction=arguments.memory_fraction,
        seed=arguments.sampling_seed,
    )
    sampled_feature_memory_shape = tuple(feature_memory.shape)
    del complete_feature_memory

    memory_seconds = time.perf_counter() - memory_start

    validation_start = time.perf_counter()
    validation_patch_scores, validation_scored_paths = (
        compute_patch_scores_for_batches(
            validation_loader,
            embedding_extractor,
            feature_memory,
            memory_chunk_size=arguments.memory_chunk_size,
            patch_grid_size=patch_grid_size,
        )
    )
    validation_seconds = time.perf_counter() - validation_start

    expected_validation_paths = tuple(str(path) for path in validation_paths)

    if validation_scored_paths != expected_validation_paths:
        raise RuntimeError(
            "Validation score paths do not match the manifest order."
        )

    test_start = time.perf_counter()
    test_patch_scores, test_scored_paths = compute_patch_scores_for_batches(
        test_loader,
        embedding_extractor,
        feature_memory,
        memory_chunk_size=arguments.memory_chunk_size,
        patch_grid_size=patch_grid_size,
    )
    test_seconds = time.perf_counter() - test_start

    expected_test_paths = tuple(str(path) for path in test_image_paths)

    if test_scored_paths != expected_test_paths:
        raise RuntimeError(
            "Test score paths do not match the discovered test-image order."
        )

    expected_test_labels = torch.tensor(
        [image.is_anomalous for image in labeled_test_images],
        dtype=torch.bool,
    )

    validation_score_rules = {
        "maximum": validation_patch_scores.flatten(start_dim=1)
        .max(dim=1)
        .values,
        "top 1 percent mean": aggregate_top_patch_scores(
            validation_patch_scores,
            top_fraction=0.01,
        ),
        "top 5 percent mean": aggregate_top_patch_scores(
            validation_patch_scores,
            top_fraction=0.05,
        ),
    }
    test_score_rules = {
        "maximum": test_patch_scores.flatten(start_dim=1).max(dim=1).values,
        "top 1 percent mean": aggregate_top_patch_scores(
            test_patch_scores,
            top_fraction=0.01,
        ),
        "top 5 percent mean": aggregate_top_patch_scores(
            test_patch_scores,
            top_fraction=0.05,
        ),
    }

    print(f"Dataset: {manifest.dataset}")
    print(f"Category: {manifest.category}")
    print(f"Input size: {arguments.input_size} x {arguments.input_size}")
    print(f"Patch grid size: {patch_grid_size}")
    print(f"Fitting images: {len(fitting_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Complete feature memory shape: {complete_feature_memory_shape}")
    print(f"Sampled feature memory shape: {sampled_feature_memory_shape}")
    print(f"Feature memory fraction: {arguments.memory_fraction:.4f}")
    print(f"Feature memory sampling seed: {arguments.sampling_seed}")
    print(
        "Feature memory build and sampling time: "
        f"{memory_seconds:.2f} seconds"
    )
    print(f"Validation scoring time: {validation_seconds:.2f} seconds")
    print(f"Test scoring time: {test_seconds:.2f} seconds")
    print()
    print(
        "Note: Aggregation and memory-sampling comparisons are exploratory. "
        "Input-size comparisons must be interpreted as follow-up experiments."
    )

    for name, validation_scores in validation_score_rules.items():
        print_score_rule_results(
            name,
            validation_scores,
            test_score_rules[name],
            test_scored_paths,
            labeled_test_images,
            expected_test_labels,
        )


if __name__ == "__main__":
    main()
