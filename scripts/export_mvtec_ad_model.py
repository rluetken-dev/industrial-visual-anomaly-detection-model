from argparse import ArgumentParser, Namespace
from pathlib import Path

from industrial_visual_anomaly_detection.datasets import (
    load_split_manifest,
    resolve_dataset_image_paths,
    validate_split_manifest,
)
from industrial_visual_anomaly_detection.training import (
    ModelTrainingConfiguration,
    train_model_artifact,
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
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--memory-chunk-size",
        type=int,
        default=4096,
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=1.0,
        help=(
            "Normal-validation score quantile used as "
            "the anomaly threshold."
        ),
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def export_model() -> None:
    arguments = parse_arguments()

    configuration = ModelTrainingConfiguration(
        batch_size=arguments.batch_size,
        memory_chunk_size=arguments.memory_chunk_size,
        input_size=arguments.input_size,
        top_fraction=arguments.top_fraction,
        threshold_quantile=arguments.threshold_quantile,
        memory_fraction=arguments.memory_fraction,
        sampling_seed=arguments.sampling_seed,
    )

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

    result = train_model_artifact(
        fitting_paths=fitting_paths,
        validation_paths=validation_paths,
        dataset=manifest.dataset,
        category=manifest.category,
        output_directory=arguments.output_directory,
        configuration=configuration,
    )

    patch_grid_side = configuration.input_size // 8
    patch_grid_size = (patch_grid_side, patch_grid_side)

    print(f"Dataset: {manifest.dataset}")
    print(f"Category: {manifest.category}")
    print(f"Fitting images: {result.fitting_image_count}")
    print(
        f"Validation images: "
        f"{result.validation_image_count}"
    )
    print(
        f"Input size: {configuration.input_size} x "
        f"{configuration.input_size}"
    )
    print(f"Patch grid size: {patch_grid_size}")
    print(
        "Complete feature memory shape: "
        f"{result.complete_memory_shape}"
    )
    print(
        "Exported feature memory shape: "
        f"{result.exported_memory_shape}"
    )
    print(
        "Feature memory size: "
        f"{result.feature_memory_size_mib:.2f} MiB"
    )
    print(
        f"Memory fraction: "
        f"{configuration.memory_fraction:.4f}"
    )
    print(f"Sampling seed: {configuration.sampling_seed}")
    print("Aggregation method: top_fraction_mean")
    print(
        f"Top fraction: "
        f"{configuration.top_fraction:.4f}"
    )
    print("Threshold method: normal_score_quantile")
    print(
        f"Threshold quantile: "
        f"{configuration.threshold_quantile:.4f}"
    )
    print(f"Validation threshold: {result.threshold:.6f}")
    print(
        "Feature memory build time: "
        f"{result.memory_seconds:.2f} seconds"
    )
    print(
        "Validation threshold calculation time: "
        f"{result.validation_seconds:.2f} seconds"
    )
    print(
        f"Artifact write time: "
        f"{result.export_seconds:.2f} seconds"
    )
    print(f"Artifact directory: {result.artifact_directory}")


def main() -> None:
    export_model()


if __name__ == "__main__":
    main()