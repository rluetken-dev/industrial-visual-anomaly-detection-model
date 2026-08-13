import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from PIL import Image


EXPECTED_CATEGORIES = {
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
}

EXPECTED_PARTITIONS = (
    "train/good",
    "validation/good",
    "test/good",
    "test/logical_anomalies",
    "test/structural_anomalies",
    "ground_truth/logical_anomalies",
    "ground_truth/structural_anomalies",
)


@dataclass(frozen=True)
class CategoryInventory:
    category: str
    train_good: int
    validation_good: int
    test_good: int
    test_logical: int
    test_structural: int
    logical_mask_groups: int
    structural_mask_groups: int
    mask_files: int


@dataclass(frozen=True)
class ImageValidationSummary:
    file_count: int
    sizes: dict[tuple[int, int], int]
    modes: dict[str, int]


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Validate a local MVTec LOCO AD dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the generated JSON validation report.",
    )
    return parser.parse_args()


def validate_structure(dataset_root: Path, categories: list[str]) -> None:
    errors: list[str] = []

    missing_categories = sorted(EXPECTED_CATEGORIES - set(categories))
    unexpected_categories = sorted(set(categories) - EXPECTED_CATEGORIES)

    if missing_categories:
        errors.append(f"Missing categories: {', '.join(missing_categories)}")

    if unexpected_categories:
        errors.append(f"Unexpected categories: {', '.join(unexpected_categories)}")

    for filename in ("readme.txt", "license.txt"):
        if not (dataset_root / filename).is_file():
            errors.append(f"Missing root file: {filename}")

    for category in categories:
        category_root = dataset_root / category

        for filename in ("readme.txt", "license.txt", "defects_config.json"):
            if not (category_root / filename).is_file():
                errors.append(f"Missing file: {category}/{filename}")

        for relative_path in EXPECTED_PARTITIONS:
            if not (category_root / relative_path).is_dir():
                errors.append(f"Missing directory: {category}/{relative_path}")

    if errors:
        raise ValueError("\n".join(errors))


def create_inventory(
    dataset_root: Path,
    categories: list[str],
) -> list[CategoryInventory]:
    inventory: list[CategoryInventory] = []

    for category in categories:
        category_root = dataset_root / category

        entry = CategoryInventory(
            category=category,
            train_good=len(
                list((category_root / "train" / "good").glob("*.png"))
            ),
            validation_good=len(
                list((category_root / "validation" / "good").glob("*.png"))
            ),
            test_good=len(
                list((category_root / "test" / "good").glob("*.png"))
            ),
            test_logical=len(
                list(
                    (
                        category_root
                        / "test"
                        / "logical_anomalies"
                    ).glob("*.png")
                )
            ),
            test_structural=len(
                list(
                    (
                        category_root
                        / "test"
                        / "structural_anomalies"
                    ).glob("*.png")
                )
            ),
            logical_mask_groups=len(
                [
                    path
                    for path in (
                        category_root
                        / "ground_truth"
                        / "logical_anomalies"
                    ).iterdir()
                    if path.is_dir()
                ]
            ),
            structural_mask_groups=len(
                [
                    path
                    for path in (
                        category_root
                        / "ground_truth"
                        / "structural_anomalies"
                    ).iterdir()
                    if path.is_dir()
                ]
            ),
            mask_files=len(
                list((category_root / "ground_truth").glob("*/*/*.png"))
            ),
        )

        if entry.test_logical != entry.logical_mask_groups:
            raise ValueError(
                f"Logical image and mask-group counts differ for {category}: "
                f"{entry.test_logical} images, "
                f"{entry.logical_mask_groups} groups"
            )

        if entry.test_structural != entry.structural_mask_groups:
            raise ValueError(
                f"Structural image and mask-group counts differ for {category}: "
                f"{entry.test_structural} images, "
                f"{entry.structural_mask_groups} groups"
            )

        inventory.append(entry)

    return inventory


def validate_image_files(dataset_root: Path) -> ImageValidationSummary:
    image_paths = sorted(dataset_root.glob("**/*.png"))
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    errors: list[str] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.load()
                sizes[image.size] += 1
                modes[image.mode] += 1
        except (OSError, ValueError) as error:
            errors.append(f"{image_path}: {error}")

    if errors:
        raise ValueError("Unreadable images:\n" + "\n".join(errors))

    return ImageValidationSummary(
        file_count=len(image_paths),
        sizes=dict(sorted(sizes.items())),
        modes=dict(sorted(modes.items())),
    )


def load_defect_pixel_values(
    dataset_root: Path,
    category: str,
) -> set[int]:
    config_path = dataset_root / category / "defects_config.json"

    with config_path.open(encoding="utf-8") as config_file:
        defect_config = json.load(config_file)

    pixel_values = {0}

    for defect in defect_config:
        pixel_value = defect.get("pixel_value")

        if not isinstance(pixel_value, int) or not 1 <= pixel_value <= 255:
            raise ValueError(
                f"Invalid pixel value in {config_path}: {pixel_value}"
            )

        pixel_values.add(pixel_value)

    return pixel_values


def validate_mask_groups(dataset_root: Path, categories: list[str]) -> tuple[int, int]:
    errors: list[str] = []
    validated_groups = 0
    validated_masks = 0

    for category in categories:
        category_root = dataset_root / category
        allowed_mask_values = load_defect_pixel_values(
            dataset_root,
            category,
        )

        for anomaly_type in ("logical_anomalies", "structural_anomalies"):
            test_root = category_root / "test" / anomaly_type
            ground_truth_root = category_root / "ground_truth" / anomaly_type

            test_images = sorted(test_root.glob("*.png"))
            expected_group_names = {image_path.stem for image_path in test_images}
            actual_group_names = {
                path.name for path in ground_truth_root.iterdir() if path.is_dir()
            }

            missing_groups = sorted(expected_group_names - actual_group_names)
            unexpected_groups = sorted(actual_group_names - expected_group_names)

            if missing_groups:
                errors.append(
                    f"Missing mask groups for {category}/{anomaly_type}: "
                    f"{', '.join(missing_groups)}"
                )

            if unexpected_groups:
                errors.append(
                    f"Unexpected mask groups for {category}/{anomaly_type}: "
                    f"{', '.join(unexpected_groups)}"
                )

            for image_path in test_images:
                mask_group = ground_truth_root / image_path.stem
                mask_paths = sorted(mask_group.glob("*.png"))

                if not mask_paths:
                    errors.append(f"Empty mask group: {mask_group}")
                    continue

                with Image.open(image_path) as image:
                    image_size = image.size

                for mask_path in mask_paths:
                    with Image.open(mask_path) as mask:
                        mask.load()

                        if mask.size != image_size:
                            errors.append(
                                f"Size mismatch: {image_path} and {mask_path}"
                            )

                        if mask.mode != "L":
                            errors.append(
                                f"Unexpected mask mode {mask.mode}: {mask_path}"
                            )

                        mask_values = {
                            value
                            for _, value in mask.getcolors(maxcolors=256) or []
                        }

                        if not mask_values.issubset(allowed_mask_values):
                            unexpected_values = sorted(
                                mask_values - allowed_mask_values
                            )
                            errors.append(
                                f"Unexpected mask values {unexpected_values}: "
                                f"{mask_path}"
                            )

                        if not any(value != 0 for value in mask_values):
                            errors.append(
                                f"Mask contains no anomaly pixels: {mask_path}"
                            )

                    validated_masks += 1

                validated_groups += 1

    if errors:
        raise ValueError("Mask validation failed:\n" + "\n".join(errors))

    return validated_groups, validated_masks


def create_report(
    dataset_root: Path,
    categories: list[str],
    inventory: list[CategoryInventory],
    image_summary: ImageValidationSummary,
    validated_mask_groups: int,
    validated_masks: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "mvtec-loco-ad",
        "dataset_root": str(dataset_root),
        "status": "passed",
        "categories": categories,
        "inventory": [
            {
                "category": item.category,
                "train_good": item.train_good,
                "validation_good": item.validation_good,
                "test_good": item.test_good,
                "test_logical": item.test_logical,
                "test_structural": item.test_structural,
                "logical_mask_groups": item.logical_mask_groups,
                "structural_mask_groups": item.structural_mask_groups,
                "mask_files": item.mask_files,
            }
            for item in inventory
        ],
        "totals": {
            "train_good": sum(item.train_good for item in inventory),
            "validation_good": sum(
                item.validation_good for item in inventory
            ),
            "test_good": sum(item.test_good for item in inventory),
            "test_logical": sum(item.test_logical for item in inventory),
            "test_structural": sum(
                item.test_structural for item in inventory
            ),
            "logical_mask_groups": sum(
                item.logical_mask_groups for item in inventory
            ),
            "structural_mask_groups": sum(
                item.structural_mask_groups for item in inventory
            ),
            "mask_files": sum(item.mask_files for item in inventory),
        },
        "images": {
            "file_count": image_summary.file_count,
            "sizes": [
                {
                    "width": width,
                    "height": height,
                    "count": count,
                }
                for (width, height), count in image_summary.sizes.items()
            ],
            "modes": image_summary.modes,
        },
        "mask_groups": {
            "validated": validated_mask_groups,
        },
        "mask_files": {
            "validated": validated_masks,
        },
        "validations": {
            "structure": "passed",
            "inventory": "passed",
            "image_readability": "passed",
            "masks": "passed",
        },
    }


def write_report(report_path: Path, report: dict[str, object]) -> None:
    resolved_report_path = report_path.resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")


def main() -> None:
    arguments = parse_arguments()
    dataset_root = arguments.dataset_root.resolve()

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {dataset_root}"
        )

    categories = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir()
    )

    validate_structure(dataset_root, categories)
    inventory = create_inventory(dataset_root, categories)
    image_summary = validate_image_files(dataset_root)
    validated_mask_groups, validated_masks = validate_mask_groups(
        dataset_root,
        categories,
    )

    if arguments.report is not None:
        report = create_report(
            dataset_root,
            categories,
            inventory,
            image_summary,
            validated_mask_groups,
            validated_masks,
        )
        write_report(arguments.report, report)

    print(f"Dataset root: {dataset_root}")
    print(f"Categories: {len(categories)}")
    print("Structure validation: passed")
    print(f"Normal training images: {sum(item.train_good for item in inventory)}")
    print(
        "Normal validation images: "
        f"{sum(item.validation_good for item in inventory)}"
    )
    print(f"Normal test images: {sum(item.test_good for item in inventory)}")
    print(
        "Logical anomaly test images: "
        f"{sum(item.test_logical for item in inventory)}"
    )
    print(
        "Structural anomaly test images: "
        f"{sum(item.test_structural for item in inventory)}"
    )
    print(
        "Logical mask groups: "
        f"{sum(item.logical_mask_groups for item in inventory)}"
    )
    print(
        "Structural mask groups: "
        f"{sum(item.structural_mask_groups for item in inventory)}"
    )
    print(f"Mask files: {sum(item.mask_files for item in inventory)}")
    print("Inventory validation: passed")
    print(f"Readable PNG files: {image_summary.file_count}")
    print(f"Distinct image sizes: {len(image_summary.sizes)}")
    print(f"Image sizes: {image_summary.sizes}")
    print(f"Image modes: {image_summary.modes}")
    print("Image readability validation: passed")
    print(f"Validated mask groups: {validated_mask_groups}")
    print(f"Validated mask files: {validated_masks}")
    print("Mask validation: passed")
    if arguments.report is not None:
        print(f"JSON report: {arguments.report.resolve()}")


if __name__ == "__main__":
    main()