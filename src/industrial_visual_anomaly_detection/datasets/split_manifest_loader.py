import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .split_manifest import DatasetSplitManifest


def validate_split_manifest(manifest: DatasetSplitManifest) -> None:
    """Validate split counts, membership, and relative path safety."""

    errors: list[str] = []

    fitting_paths = set(manifest.fitting_images)
    validation_paths = set(manifest.validation_images)

    if len(manifest.fitting_images) != manifest.fitting_image_count:
        errors.append(
            "Fitting image count does not match the manifest entries."
        )

    if len(manifest.validation_images) != manifest.validation_image_count:
        errors.append(
            "Validation image count does not match the manifest entries."
        )

    if (
        len(manifest.fitting_images) + len(manifest.validation_images)
        != manifest.source_image_count
    ):
        errors.append(
            "Fitting and validation counts do not match the source image count."
        )

    if len(fitting_paths) != len(manifest.fitting_images):
        errors.append("Fitting partition contains duplicate paths.")

    if len(validation_paths) != len(manifest.validation_images):
        errors.append("Validation partition contains duplicate paths.")

    overlap = fitting_paths & validation_paths

    if overlap:
        errors.append(
            f"Fitting and validation partitions overlap in {len(overlap)} path(s)."
        )

    all_paths = fitting_paths | validation_paths

    if any(
        path.is_absolute()
        or PurePosixPath(path.as_posix()).is_absolute()
        or PureWindowsPath(str(path)).is_absolute()
        for path in all_paths
    ):
        errors.append("Manifest image paths must be relative.")

    if any(".." in path.parts for path in all_paths):
        errors.append("Manifest image paths must not contain parent traversal.")

    if errors:
        raise ValueError("\n".join(errors))


def load_split_manifest(manifest_path: Path) -> DatasetSplitManifest:
    """Load a deterministic dataset split manifest from JSON."""

    resolved_manifest_path = manifest_path.resolve()

    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest does not exist: {resolved_manifest_path}"
        )

    with resolved_manifest_path.open(encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)

    manifest = DatasetSplitManifest(
        schema_version=int(data["schema_version"]),
        dataset=str(data["dataset"]),
        category=str(data["category"]),
        seed=int(data["seed"]),
        source_image_count=int(data["source_image_count"]),
        fitting_image_count=int(data["fitting_image_count"]),
        validation_image_count=int(data["validation_image_count"]),
        fitting_images=tuple(
            Path(relative_path)
            for relative_path in data["fitting_images"]
        ),
        validation_images=tuple(
            Path(relative_path)
            for relative_path in data["validation_images"]
        ),
    )

    validate_split_manifest(manifest)

    return manifest