from argparse import ArgumentParser, Namespace
from pathlib import Path

from industrial_visual_anomaly_detection.datasets import (
    create_image_path_split,
    discover_image_paths,
    save_image_path_split_manifest,
)
from industrial_visual_anomaly_detection.training import (
    ModelTrainingConfiguration,
    train_model_artifact,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Train and export an anomaly-detection model artifact "
            "from a directory containing normal images."
        )
    )
    parser.add_argument(
        "--image-directory",
        required=True,
        type=Path,
        help=(
            "Directory containing normal PNG or JPEG images. "
            "Subdirectories are searched recursively."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name stored in the artifact metadata.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Category name stored in the artifact metadata.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument("--split-seed", type=int, default=42)
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


def main() -> None:
    arguments = parse_arguments()

    image_paths = discover_image_paths(
        arguments.image_directory
    )
    split = create_image_path_split(
        image_paths,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.split_seed,
    )

    configuration = ModelTrainingConfiguration(
        batch_size=arguments.batch_size,
        memory_chunk_size=arguments.memory_chunk_size,
        input_size=arguments.input_size,
        top_fraction=arguments.top_fraction,
        memory_fraction=arguments.memory_fraction,
        sampling_seed=arguments.sampling_seed,
    )

    result = train_model_artifact(
        fitting_paths=split.fitting_paths,
        validation_paths=split.validation_paths,
        dataset=arguments.dataset,
        category=arguments.category,
        output_directory=arguments.output_directory,
        configuration=configuration,
    )

    split_manifest_path = save_image_path_split_manifest(
        image_directory=arguments.image_directory,
        split=split,
        dataset=arguments.dataset,
        category=arguments.category,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.split_seed,
        output_path=(
            result.artifact_directory / "training_split.json"
        ),
    )

    patch_grid_side = configuration.input_size // 8
    patch_grid_size = (patch_grid_side, patch_grid_side)

    print(f"Dataset: {arguments.dataset}")
    print(f"Category: {arguments.category}")
    print(
        f"Image directory: "
        f"{arguments.image_directory.resolve()}"
    )
    print(f"Source images: {len(image_paths)}")
    print(f"Split seed: {arguments.split_seed}")
    print(
        f"Validation fraction: "
        f"{arguments.validation_fraction:.4f}"
    )
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
    print(f"Training split manifest: {split_manifest_path}")


if __name__ == "__main__":
    main()