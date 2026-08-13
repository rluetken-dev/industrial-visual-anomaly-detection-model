import tempfile
import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets import (
    discover_mvtec_ad_test_images,
)


class MvtecAdTestDiscoveryTests(unittest.TestCase):
    def test_images_are_discovered_with_labels_and_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            test_root = dataset_root / "bottle" / "test"

            good_directory = test_root / "good"
            defect_directory = test_root / "broken_small"

            good_directory.mkdir(parents=True)
            defect_directory.mkdir(parents=True)

            (good_directory / "001.png").touch()
            (good_directory / "000.png").touch()
            (defect_directory / "000.png").touch()

            images = discover_mvtec_ad_test_images(
                dataset_root,
                "bottle",
            )

            self.assertEqual(3, len(images))
            self.assertEqual(
                [
                    ("broken_small", True, "000.png"),
                    ("good", False, "000.png"),
                    ("good", False, "001.png"),
                ],
                [
                    (
                        image.group,
                        image.is_anomalous,
                        image.path.name,
                    )
                    for image in images
                ],
            )

    def test_missing_test_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                NotADirectoryError,
                "Test directory does not exist",
            ):
                discover_mvtec_ad_test_images(
                    Path(temporary_directory),
                    "bottle",
                )


if __name__ == "__main__":
    unittest.main()