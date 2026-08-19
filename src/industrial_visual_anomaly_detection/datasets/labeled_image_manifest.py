import csv
from pathlib import Path

from .image_discovery import SUPPORTED_IMAGE_SUFFIXES
from .labeled_image import LabeledImage


REQUIRED_COLUMNS = frozenset(
    {
        "image",
        "group",
        "is_anomalous",
    }
)


def load_labeled_image_manifest(
    dataset_root: Path,
    manifest_path: Path,
) -> tuple[LabeledImage, ...]:
    """Load labeled images from a dataset-independent CSV manifest."""

    resolved_dataset_root = dataset_root.resolve()
    resolved_manifest_path = manifest_path.resolve()

    if not resolved_dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: "
            f"{resolved_dataset_root}"
        )

    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(
            f"Labeled-image manifest does not exist: "
            f"{resolved_manifest_path}"
        )

    labeled_images: list[LabeledImage] = []
    discovered_paths: set[Path] = set()

    with resolved_manifest_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            formatted_columns = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                "Labeled-image manifest is missing required "
                f"columns: {formatted_columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            image_value = row["image"].strip()
            group = row["group"].strip()
            anomaly_value = row["is_anomalous"].strip().lower()

            if not image_value:
                raise ValueError(
                    "Labeled-image manifest contains no image "
                    f"path at row {row_number}."
                )

            if not group:
                raise ValueError(
                    "Labeled-image manifest contains no group "
                    f"at row {row_number}."
                )

            if anomaly_value not in {"true", "false"}:
                raise ValueError(
                    "Labeled-image manifest contains an invalid "
                    "is_anomalous value at row "
                    f"{row_number}: {anomaly_value}"
                )

            image_path = (
                resolved_dataset_root / Path(image_value)
            ).resolve()

            if not image_path.is_relative_to(
                resolved_dataset_root
            ):
                raise ValueError(
                    "Labeled image is outside the dataset "
                    f"directory at row {row_number}: {image_path}"
                )

            if image_path.suffix.lower() not in (
                SUPPORTED_IMAGE_SUFFIXES
            ):
                raise ValueError(
                    "Labeled image has an unsupported file "
                    f"extension at row {row_number}: {image_path}"
                )

            if not image_path.is_file():
                raise FileNotFoundError(
                    "Labeled image does not exist at row "
                    f"{row_number}: {image_path}"
                )

            if image_path in discovered_paths:
                raise ValueError(
                    "Labeled-image manifest contains a duplicate "
                    f"image path: {image_path}"
                )

            discovered_paths.add(image_path)
            labeled_images.append(
                LabeledImage(
                    path=image_path,
                    group=group,
                    is_anomalous=anomaly_value == "true",
                )
            )

    if not labeled_images:
        raise ValueError(
            "Labeled-image manifest contains no images."
        )

    return tuple(labeled_images)