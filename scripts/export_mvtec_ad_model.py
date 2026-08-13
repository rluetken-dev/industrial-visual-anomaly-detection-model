import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from torch.utils.data import DataLoader

from industrial_visual_anomaly_detection.artifacts import (
    ModelArtifact,
    ModelArtifactMetadata,
    save_model_artifact,
)
from industrial_visual_anomaly_detection.datasets import (
    ImagePathDataset,
    load_split_manifest,
    resolve_dataset_image_paths,
    validate_split_manifest,
)
from industrial_visual_anomaly_detection.evaluation import (
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
            "Export a reusable anomaly-detection model artifact "
            "for one MVTec AD category."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=42)
    return parser.parse_args()


def validate_arguments(arguments: Namespace) -> None:
    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if arguments.memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    if arguments.input_size <= 0:
        raise ValueError("Input size must be greater than zero.")

    if arguments.input_size % 32 != 0:
        raise ValueError("Input size must be divisible by 32.")

    if not 0.0 < arguments.top_fraction <= 1.0:
        raise ValueError(
            "Top fraction must be greater than zero and at most one."
        )

    if not 0.0 < arguments.memory_fraction <= 1.0:
        raise ValueError(
            "Memory fraction must be greater than zero and at most one."
        )


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

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

    preprocessing = create_image_preprocessing(
        input_size=(arguments.input_size, arguments.input_size)
    )

    fitting_dataset = ImagePathDataset(fitting_paths, preprocessing)
    validation_dataset = ImagePathDataset(
        validation_paths,
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

    embedding_extractor = create_resnet18_patch_embedding_extractor()

    memory_start = time.perf_counter()
    complete_feature_memory = build_feature_memory(
        fitting_loader,
        embedding_extractor,
    )
    complete_memory_shape = tuple(complete_feature_memory.shape)

    feature_memory = sample_feature_memory(
        complete_feature_memory,
        fraction=arguments.memory_fraction,
        seed=arguments.sampling_seed,
    )
    feature_memory = feature_memory.contiguous()
    exported_memory_shape = tuple(feature_memory.shape)
    del complete_feature_memory

    memory_seconds = time.perf_counter() - memory_start

    patch_grid_side = arguments.input_size // 8
    patch_grid_size = (patch_grid_side, patch_grid_side)

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

    expected_validation_paths = tuple(
        str(path) for path in validation_paths
    )

    if validation_scored_paths != expected_validation_paths:
        raise RuntimeError(
            "Validation score paths do not match the manifest order."
        )

    validation_scores = aggregate_top_patch_scores(
        validation_patch_scores,
        top_fraction=arguments.top_fraction,
    )
    threshold = select_maximum_normal_threshold(validation_scores)

    metadata = ModelArtifactMetadata(
        schema_version=1,
        dataset=manifest.dataset,
        category=manifest.category,
        backbone="resnet18",
        input_size=arguments.input_size,
        patch_grid_size=patch_grid_size,
        embedding_dimension=feature_memory.shape[1],
        aggregation_method="top_fraction_mean",
        top_fraction=arguments.top_fraction,
        threshold=threshold,
        memory_fraction=arguments.memory_fraction,
        sampling_seed=arguments.sampling_seed,
        feature_memory_entries=feature_memory.shape[0],
    )
    artifact = ModelArtifact(
        metadata=metadata,
        feature_memory=feature_memory,
    )

    export_start = time.perf_counter()
    artifact_directory = save_model_artifact(
        artifact,
        arguments.output_directory,
    )
    export_seconds = time.perf_counter() - export_start

    feature_memory_size_mib = (
        feature_memory.numel()
        * feature_memory.element_size()
        / (1024 * 1024)
    )

    print(f"Dataset: {manifest.dataset}")
    print(f"Category: {manifest.category}")
    print(f"Fitting images: {len(fitting_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Input size: {arguments.input_size} x {arguments.input_size}")
    print(f"Patch grid size: {patch_grid_size}")
    print(f"Complete feature memory shape: {complete_memory_shape}")
    print(f"Exported feature memory shape: {exported_memory_shape}")
    print(f"Feature memory size: {feature_memory_size_mib:.2f} MiB")
    print(f"Memory fraction: {arguments.memory_fraction:.4f}")
    print(f"Sampling seed: {arguments.sampling_seed}")
    print(f"Aggregation method: {metadata.aggregation_method}")
    print(f"Top fraction: {metadata.top_fraction:.4f}")
    print(f"Validation threshold: {metadata.threshold:.6f}")
    print(f"Feature memory build time: {memory_seconds:.2f} seconds")
    print(
        "Validation threshold calculation time: "
        f"{validation_seconds:.2f} seconds"
    )
    print(f"Artifact write time: {export_seconds:.2f} seconds")
    print(f"Artifact directory: {artifact_directory}")


if __name__ == "__main__":
    main()
