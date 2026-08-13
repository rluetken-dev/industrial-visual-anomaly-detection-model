import json
from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


EXPECTED_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


@dataclass(frozen=True)
class CategoryInventory:
    category: str
    train_good: int
    test_good: int
    test_anomalous: int
    masks: int


@dataclass(frozen=True)
class ImageValidationSummary:
    file_count: int
    sizes: dict[tuple[int, int], int]
    modes: dict[str, int]


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Validate a local MVTec AD dataset.")
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

        for relative_path in ("train/good", "test/good", "ground_truth"):
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
        defect_directories = sorted(
            path
            for path in (category_root / "test").iterdir()
            if path.is_dir() and path.name != "good"
        )

        entry = CategoryInventory(
            category=category,
            train_good=len(
                list((category_root / "train" / "good").glob("*.png"))
            ),
            test_good=len(
                list((category_root / "test" / "good").glob("*.png"))
            ),
            test_anomalous=sum(
                len(list(path.glob("*.png")))
                for path in defect_directories
            ),
            masks=len(
                list((category_root / "ground_truth").glob("*/*.png"))
            ),
        )

        if entry.test_anomalous != entry.masks:
            raise ValueError(
                f"Anomaly and mask counts differ for {category}: "
                f"{entry.test_anomalous} images, {entry.masks} masks"
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


def validate_mask_pairs(dataset_root: Path, categories: list[str]) -> int:
    errors: list[str] = []
    validated_pairs = 0

    for category in categories:
        category_root = dataset_root / category
        test_root = category_root / "test"

        defect_directories = sorted(
            path
            for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )

        for defect_directory in defect_directories:
            for image_path in sorted(defect_directory.glob("*.png")):
                mask_path = (
                    category_root
                    / "ground_truth"
                    / defect_directory.name
                    / f"{image_path.stem}_mask.png"
                )

                if not mask_path.is_file():
                    errors.append(f"Missing mask for: {image_path}")
                    continue

                with (
                    Image.open(image_path) as image,
                    Image.open(mask_path) as mask,
                ):
                    if image.size != mask.size:
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

                    if not mask_values.issubset({0, 255}):
                        errors.append(
                            f"Non-binary mask values {mask_values}: {mask_path}"
                        )

                    if 255 not in mask_values:
                        errors.append(f"Mask contains no anomaly pixels: {mask_path}")

                validated_pairs += 1

    if errors:
        raise ValueError("Mask validation failed:\n" + "\n".join(errors))

    return validated_pairs


def create_report(
    dataset_root: Path,
    categories: list[str],
    inventory: list[CategoryInventory],
    image_summary: ImageValidationSummary,
    validated_mask_pairs: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "mvtec-ad",
        "dataset_root": str(dataset_root),
        "status": "passed",
        "categories": categories,
        "inventory": [
            {
                "category": item.category,
                "train_good": item.train_good,
                "test_good": item.test_good,
                "test_anomalous": item.test_anomalous,
                "masks": item.masks,
            }
            for item in inventory
        ],
        "totals": {
            "train_good": sum(item.train_good for item in inventory),
            "test_good": sum(item.test_good for item in inventory),
            "test_anomalous": sum(item.test_anomalous for item in inventory),
            "masks": sum(item.masks for item in inventory),
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
        "mask_pairs": {
            "validated": validated_mask_pairs,
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
    validated_mask_pairs = validate_mask_pairs(dataset_root, categories)

    if arguments.report is not None:
        report = create_report(
            dataset_root,
            categories,
            inventory,
            image_summary,
            validated_mask_pairs,
        )
        write_report(arguments.report, report)

    print(f"Dataset root: {dataset_root}")
    print(f"Categories: {len(categories)}")

    for category in categories:
        print(f"- {category}")

    print("Structure validation: passed")
    print(
        "Normal training images: "
        f"{sum(item.train_good for item in inventory)}"
    )
    print(
        "Normal test images: "
        f"{sum(item.test_good for item in inventory)}"
    )
    print(
        "Anomalous test images: "
        f"{sum(item.test_anomalous for item in inventory)}"
    )
    print(
        "Ground-truth masks: "
        f"{sum(item.masks for item in inventory)}"
    )
    print("Inventory validation: passed")
    print(f"Readable PNG files: {image_summary.file_count}")
    print(f"Distinct image sizes: {len(image_summary.sizes)}")
    print(f"Image sizes: {image_summary.sizes}")
    print(f"Image modes: {image_summary.modes}")
    print("Image readability validation: passed")
    print(f"Validated image-mask pairs: {validated_mask_pairs}")
    print("Mask validation: passed")
    if arguments.report is not None:
        print(f"JSON report: {arguments.report.resolve()}")


if __name__ == "__main__":
    main()