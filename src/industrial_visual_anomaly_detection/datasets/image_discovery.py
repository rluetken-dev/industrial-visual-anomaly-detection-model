from pathlib import Path


SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def discover_image_paths(
    image_directory: Path,
) -> tuple[Path, ...]:
    """Discover supported image files recursively in a directory."""

    resolved_directory = image_directory.resolve()

    if not resolved_directory.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {resolved_directory}"
        )

    image_paths = tuple(
        sorted(
            path.resolve()
            for path in resolved_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    )

    if not image_paths:
        raise ValueError(
            f"Image directory contains no supported images: "
            f"{resolved_directory}"
        )

    return image_paths