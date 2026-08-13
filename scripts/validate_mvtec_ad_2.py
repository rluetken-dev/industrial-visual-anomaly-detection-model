import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from PIL import Image


EXPECTED_CATEGORIES = {
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
}

EXPECTED_PARTITIONS = (
    "train/good",
    "validation/good",
    "test_public/good",
    "test_public/bad",
    "test_public/ground_truth/bad",
    "test_private",
    "test_private_mixed",
)


@dataclass(frozen=True)
class CategoryInventory:
    category: str
    train_good: int
    validation_good: int
    public_good: int
    public_bad: int
    public_masks: int
    private: int
    private_mixed: int


@dataclass(frozen=True)
class ImageValidationSummary:
    file_count: int
    sizes: dict[tuple[int, int], int]
    modes: dict[str, int]


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Validate a local MVTec AD 2 dataset.")
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
            public_good=len(
                list((category_root / "test_public" / "good").glob("*.png"))
            ),
            public_bad=len(
                list((category_root / "test_public" / "bad").glob("*.png"))
            ),
            public_masks=len(
                list(
                    (
                        category_root
                        / "test_public"
                        / "ground_truth"
                        / "bad"
                    ).glob("*.png")
                )
            ),
            private=len(
                list((category_root / "test_private").glob("*.png"))
            ),
            private_mixed=len(
                list((category_root / "test_private_mixed").glob("*.png"))
            ),
        )

        if entry.public_bad != entry.public_masks:
            raise ValueError(
                f"Public anomaly and mask counts differ for {category}: "
                f"{entry.public_bad} images, {entry.public_masks} masks"
            )

        inventory.append(entry)

    return inventory


def validate_mask_names(dataset_root: Path, categories: list[str]) -> int:
    errors: list[str] = []
    validated_pairs = 0

    for category in categories:
        category_root = dataset_root / category
        image_root = category_root / "test_public" / "bad"
        mask_root = category_root / "test_public" / "ground_truth" / "bad"

        expected_mask_names = {
            f"{image_path.stem}_mask.png"
            for image_path in image_root.glob("*.png")
        }
        actual_mask_names = {
            mask_path.name
            for mask_path in mask_root.glob("*.png")
        }

        missing_masks = sorted(expected_mask_names - actual_mask_names)
        unexpected_masks = sorted(actual_mask_names - expected_mask_names)

        if missing_masks:
            errors.append(
                f"Missing masks for {category}: {', '.join(missing_masks)}"
            )

        if unexpected_masks:
            errors.append(
                f"Unexpected masks for {category}: "
                f"{', '.join(unexpected_masks)}"
            )

        validated_pairs += len(expected_mask_names)

    if errors:
        raise ValueError("Mask-name validation failed:\n" + "\n".join(errors))

    return validated_pairs


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


def validate_mask_pairs(
    dataset_root: Path,
    categories: list[str],
) -> int:
    errors: list[str] = []
    validated_pairs = 0

    for category in categories:
        category_root = dataset_root / category
        image_root = category_root / "test_public" / "bad"
        mask_root = (
            category_root
            / "test_public"
            / "ground_truth"
            / "bad"
        )

        for image_path in sorted(image_root.glob("*.png")):
            mask_path = mask_root / f"{image_path.stem}_mask.png"

            if not mask_path.is_file():
                errors.append(f"Missing mask for: {image_path}")
                continue

            with (
                Image.open(image_path) as image,
                Image.open(mask_path) as mask,
            ):
                image.load()
                mask.load()

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
                    errors.append(
                        f"Mask contains no anomaly pixels: {mask_path}"
                    )

            validated_pairs += 1

    if errors:
        raise ValueError(
            "Mask-content validation failed:\n" + "\n".join(errors)
        )

    return validated_pairs


def create_report(
    dataset_root: Path,
    categories: list[str],
    inventory: list[CategoryInventory],
    image_summary: ImageValidationSummary,
    validated_mask_names: int,
    validated_mask_pairs: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "mvtec-ad-2",
        "dataset_root": str(dataset_root),
        "status": "passed",
        "categories": categories,
        "inventory": [
            {
                "category": item.category,
                "train_good": item.train_good,
                "validation_good": item.validation_good,
                "public_good": item.public_good,
                "public_bad": item.public_bad,
                "public_masks": item.public_masks,
                "private": item.private,
                "private_mixed": item.private_mixed,
            }
            for item in inventory
        ],
        "totals": {
            "train_good": sum(item.train_good for item in inventory),
            "validation_good": sum(
                item.validation_good for item in inventory
            ),
            "public_good": sum(item.public_good for item in inventory),
            "public_bad": sum(item.public_bad for item in inventory),
            "public_masks": sum(item.public_masks for item in inventory),
            "private": sum(item.private for item in inventory),
            "private_mixed": sum(
                item.private_mixed for item in inventory
            ),
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
        "public_mask_names": {
            "validated": validated_mask_names,
        },
        "public_mask_pairs": {
            "validated": validated_mask_pairs,
        },
        "validations": {
            "structure": "passed",
            "inventory": "passed",
            "mask_names": "passed",
            "image_readability": "passed",
            "mask_content": "passed",
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
    validated_pairs = validate_mask_names(dataset_root, categories)
    image_summary = validate_image_files(dataset_root)
    validated_mask_pairs = validate_mask_pairs(
        dataset_root,
        categories,
    )

    if arguments.report is not None:
        report = create_report(
            dataset_root,
            categories,
            inventory,
            image_summary,
            validated_pairs,
            validated_mask_pairs,
        )
        write_report(arguments.report, report)

    print(f"Dataset root: {dataset_root}")
    print(f"Categories: {len(categories)}")
    print("Structure validation: passed")
    print(f"Training images: {sum(item.train_good for item in inventory)}")
    print(
        "Validation images: "
        f"{sum(item.validation_good for item in inventory)}"
    )
    print(
        "Public normal test images: "
        f"{sum(item.public_good for item in inventory)}"
    )
    print(
        "Public anomalous test images: "
        f"{sum(item.public_bad for item in inventory)}"
    )
    print(
        "Public masks: "
        f"{sum(item.public_masks for item in inventory)}"
    )
    print(f"Private test images: {sum(item.private for item in inventory)}")
    print(
        "Private mixed test images: "
        f"{sum(item.private_mixed for item in inventory)}"
    )
    print("Inventory validation: passed")
    print(f"Validated public image-mask names: {validated_pairs}")
    print("Mask-name validation: passed")
    print(f"Readable PNG files: {image_summary.file_count}")
    print(f"Distinct image sizes: {len(image_summary.sizes)}")
    print(f"Image sizes: {image_summary.sizes}")
    print(f"Image modes: {image_summary.modes}")
    print("Image readability validation: passed")
    print(f"Validated public image-mask pairs: {validated_mask_pairs}")
    print("Mask-content validation: passed")
    if arguments.report is not None:
        print(f"JSON report: {arguments.report.resolve()}")


if __name__ == "__main__":
    main()