import tempfile
import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets.image_discovery import (
    discover_image_paths,
)


class ImageDiscoveryTests(unittest.TestCase):
    def test_supported_images_are_discovered_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested_directory = root / "nested"
            nested_directory.mkdir()

            expected_paths = (
                root / "a.png",
                root / "b.JPG",
                nested_directory / "c.jpeg",
            )

            for image_path in expected_paths:
                image_path.touch()

            (root / "ignored.txt").touch()

            discovered_paths = discover_image_paths(root)

            self.assertEqual(
                tuple(sorted(path.resolve() for path in expected_paths)),
                discovered_paths,
            )

    def test_missing_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory) / "missing"

            with self.assertRaisesRegex(
                NotADirectoryError,
                "Image directory does not exist",
            ):
                discover_image_paths(missing_directory)

    def test_directory_without_supported_images_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "notes.txt").touch()

            with self.assertRaisesRegex(
                ValueError,
                "Image directory contains no supported images",
            ):
                discover_image_paths(root)


if __name__ == "__main__":
    unittest.main()