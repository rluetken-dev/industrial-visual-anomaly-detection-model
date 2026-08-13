import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets import (
    DatasetSplitManifest,
    validate_split_manifest,
)


class SplitManifestValidationTests(unittest.TestCase):
    def create_valid_manifest(self) -> DatasetSplitManifest:
        return DatasetSplitManifest(
            schema_version=1,
            dataset="mvtec-ad",
            category="bottle",
            seed=42,
            source_image_count=3,
            fitting_image_count=2,
            validation_image_count=1,
            fitting_images=(
                Path("bottle/train/good/000.png"),
                Path("bottle/train/good/001.png"),
            ),
            validation_images=(
                Path("bottle/train/good/002.png"),
            ),
        )

    def test_valid_manifest_passes(self) -> None:
        validate_split_manifest(self.create_valid_manifest())

    def test_overlapping_partitions_are_rejected(self) -> None:
        manifest = DatasetSplitManifest(
            schema_version=1,
            dataset="mvtec-ad",
            category="bottle",
            seed=42,
            source_image_count=4,
            fitting_image_count=2,
            validation_image_count=2,
            fitting_images=(
                Path("bottle/train/good/000.png"),
                Path("bottle/train/good/001.png"),
            ),
            validation_images=(
                Path("bottle/train/good/001.png"),
                Path("bottle/train/good/002.png"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_manifest(manifest)

    def test_absolute_paths_are_rejected(self) -> None:
        manifest = self.create_valid_manifest()

        invalid_manifest = DatasetSplitManifest(
            schema_version=manifest.schema_version,
            dataset=manifest.dataset,
            category=manifest.category,
            seed=manifest.seed,
            source_image_count=manifest.source_image_count,
            fitting_image_count=manifest.fitting_image_count,
            validation_image_count=manifest.validation_image_count,
            fitting_images=(
                Path("C:/private/000.png"),
                manifest.fitting_images[1],
            ),
            validation_images=manifest.validation_images,
        )

        with self.assertRaisesRegex(ValueError, "relative"):
            validate_split_manifest(invalid_manifest)


if __name__ == "__main__":
    unittest.main()