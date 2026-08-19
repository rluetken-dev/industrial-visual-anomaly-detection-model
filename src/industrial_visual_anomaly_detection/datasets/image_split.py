from dataclasses import dataclass
from pathlib import Path
from random import Random


@dataclass(frozen=True)
class ImagePathSplit:
    """Contain deterministic fitting and validation image partitions."""

    fitting_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]


def create_image_path_split(
    image_paths: tuple[Path, ...],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> ImagePathSplit:
    """Split image paths deterministically into fitting and validation sets."""

    if len(image_paths) < 2:
        raise ValueError(
            "Image splitting requires at least two image paths."
        )

    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Image paths must not contain duplicates.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "Validation fraction must be greater than zero "
            "and less than one."
        )

    shuffled_paths = list(image_paths)
    Random(seed).shuffle(shuffled_paths)

    validation_count = round(
        len(shuffled_paths) * validation_fraction
    )
    validation_count = max(1, validation_count)
    validation_count = min(
        validation_count,
        len(shuffled_paths) - 1,
    )

    validation_paths = tuple(
        sorted(shuffled_paths[:validation_count])
    )
    fitting_paths = tuple(
        sorted(shuffled_paths[validation_count:])
    )

    if set(fitting_paths) & set(validation_paths):
        raise RuntimeError(
            "Fitting and validation image partitions overlap."
        )

    if len(fitting_paths) + len(validation_paths) != len(image_paths):
        raise RuntimeError(
            "Image partitions do not cover every source image."
        )

    return ImagePathSplit(
        fitting_paths=fitting_paths,
        validation_paths=validation_paths,
    )