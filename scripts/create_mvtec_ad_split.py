from argparse import ArgumentParser, Namespace
from json import dumps
from pathlib import Path
from random import Random


VALIDATION_RATIO = 0.2
SEED = 42


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Create a deterministic fitting and validation split "
            "for one MVTec AD category."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--expected-image-count",
        required=True,
        type=int,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def create_relative_paths(
    dataset_root: Path,
    image_paths: list[Path],
) -> list[str]:
    return [
        image_path.relative_to(dataset_root).as_posix()
        for image_path in image_paths
    ]


def main() -> None:
    arguments = parse_arguments()

    if arguments.expected_image_count <= 0:
        raise ValueError(
            "Expected image count must be greater than zero."
        )

    dataset_root = arguments.dataset_root.resolve()
    output_path = arguments.output.resolve()

    source_directory = (
        dataset_root
        / arguments.category
        / "train"
        / "good"
    )

    if not source_directory.is_dir():
        raise NotADirectoryError(
            f"Training directory does not exist: {source_directory}"
        )

    image_paths = sorted(source_directory.glob("*.png"))

    if len(image_paths) != arguments.expected_image_count:
        raise ValueError(
            f"Expected {arguments.expected_image_count} images, "
            f"found {len(image_paths)}."
        )

    shuffled_paths = image_paths.copy()
    Random(SEED).shuffle(shuffled_paths)

    validation_count = round(
        len(shuffled_paths) * VALIDATION_RATIO
    )
    validation_paths = sorted(
        shuffled_paths[:validation_count]
    )
    fitting_paths = sorted(
        shuffled_paths[validation_count:]
    )

    if set(fitting_paths) & set(validation_paths):
        raise ValueError(
            "Fitting and validation splits overlap."
        )

    if (
        len(fitting_paths) + len(validation_paths)
        != len(image_paths)
    ):
        raise ValueError(
            "The generated split does not cover every source image."
        )

    manifest = {
        "schema_version": 1,
        "dataset": "mvtec-ad",
        "category": arguments.category,
        "source_partition": "train/good",
        "seed": SEED,
        "fitting_ratio": 1.0 - VALIDATION_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "source_image_count": len(image_paths),
        "fitting_image_count": len(fitting_paths),
        "validation_image_count": len(validation_paths),
        "fitting_images": create_relative_paths(
            dataset_root,
            fitting_paths,
        ),
        "validation_images": create_relative_paths(
            dataset_root,
            validation_paths,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: mvtec-ad")
    print(f"Category: {arguments.category}")
    print(f"Seed: {SEED}")
    print(f"Source images: {len(image_paths)}")
    print(f"Fitting images: {len(fitting_paths)}")
    print(f"Validation images: {len(validation_paths)}")
    print("Overlap: 0")
    print(f"Manifest written to: {output_path}")


if __name__ == "__main__":
    main()