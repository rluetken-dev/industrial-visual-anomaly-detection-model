import json
from pathlib import Path

from .image_split import ImagePathSplit


def save_image_path_split_manifest(
    image_directory: Path,
    split: ImagePathSplit,
    dataset: str,
    category: str,
    validation_fraction: float,
    seed: int,
    output_path: Path,
) -> Path:
    """Save a reproducible fitting and validation split manifest."""

    resolved_image_directory = image_directory.resolve()

    if not resolved_image_directory.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: "
            f"{resolved_image_directory}"
        )

    if not dataset.strip():
        raise ValueError("Dataset name is required.")

    if not category.strip():
        raise ValueError("Category name is required.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "Validation fraction must be greater than zero "
            "and less than one."
        )

    fitting_images = _create_relative_paths(
        resolved_image_directory,
        split.fitting_paths,
    )
    validation_images = _create_relative_paths(
        resolved_image_directory,
        split.validation_paths,
    )

    source_image_count = (
        len(fitting_images) + len(validation_images)
    )

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "category": category,
        "source_directory": ".",
        "seed": seed,
        "fitting_ratio": 1.0 - validation_fraction,
        "validation_ratio": validation_fraction,
        "source_image_count": source_image_count,
        "fitting_image_count": len(fitting_images),
        "validation_image_count": len(validation_images),
        "fitting_images": fitting_images,
        "validation_images": validation_images,
    }

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return resolved_output_path


def _create_relative_paths(
    image_directory: Path,
    image_paths: tuple[Path, ...],
) -> list[str]:
    relative_paths: list[str] = []

    for image_path in image_paths:
        resolved_image_path = image_path.resolve()

        if not resolved_image_path.is_relative_to(
            image_directory
        ):
            raise ValueError(
                "Split image path is outside the image directory: "
                f"{resolved_image_path}"
            )

        relative_paths.append(
            resolved_image_path.relative_to(
                image_directory
            ).as_posix()
        )

    return relative_paths