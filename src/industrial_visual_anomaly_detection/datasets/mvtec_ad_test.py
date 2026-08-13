from pathlib import Path

from .labeled_image import LabeledImage


def discover_mvtec_ad_test_images(
    dataset_root: Path,
    category: str,
) -> tuple[LabeledImage, ...]:
    """Discover labeled MVTec AD test images for one category."""

    resolved_dataset_root = dataset_root.resolve()
    test_root = resolved_dataset_root / category / "test"

    if not test_root.is_dir():
        raise NotADirectoryError(
            f"Test directory does not exist: {test_root}"
        )

    group_directories = sorted(
        path for path in test_root.iterdir() if path.is_dir()
    )

    if not group_directories:
        raise ValueError(f"No test groups found: {test_root}")

    labeled_images: list[LabeledImage] = []

    for group_directory in group_directories:
        image_paths = sorted(group_directory.glob("*.png"))

        if not image_paths:
            raise ValueError(
                f"Test group contains no PNG images: {group_directory}"
            )

        for image_path in image_paths:
            labeled_images.append(
                LabeledImage(
                    path=image_path.resolve(),
                    group=group_directory.name,
                    is_anomalous=group_directory.name != "good",
                )
            )

    return tuple(labeled_images)