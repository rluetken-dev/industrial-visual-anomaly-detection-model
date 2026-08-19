import tempfile
import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets.labeled_image_manifest import (
    load_labeled_image_manifest,
)


class LabeledImageManifestTests(unittest.TestCase):
    def test_labeled_images_are_loaded_in_manifest_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            normal_image = dataset_root / "normal.JPG"
            anomaly_image = dataset_root / "anomaly.png"
            normal_image.touch()
            anomaly_image.touch()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "normal.JPG,normal,false\n"
                "anomaly.png,scratch,true\n",
                encoding="utf-8",
            )

            images = load_labeled_image_manifest(
                dataset_root,
                manifest_path,
            )

            self.assertEqual(2, len(images))

            self.assertEqual(
                normal_image.resolve(),
                images[0].path,
            )
            self.assertEqual("normal", images[0].group)
            self.assertFalse(images[0].is_anomalous)

            self.assertEqual(
                anomaly_image.resolve(),
                images[1].path,
            )
            self.assertEqual("scratch", images[1].group)
            self.assertTrue(images[1].is_anomalous)

    def test_missing_dataset_directory_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n",
                encoding="utf-8",
            )

            with self.assertRaises(NotADirectoryError):
                load_labeled_image_manifest(
                    temporary_path / "missing",
                    manifest_path,
                )

    def test_missing_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory) / "dataset"
            dataset_root.mkdir()

            with self.assertRaises(FileNotFoundError):
                load_labeled_image_manifest(
                    dataset_root,
                    dataset_root / "missing.csv",
                )

    def test_missing_required_column_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group\n"
                "image.png,normal\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "is_anomalous",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_invalid_anomaly_value_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            image_path = dataset_root / "image.png"
            image_path.touch()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "image.png,normal,yes\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "invalid is_anomalous",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_image_outside_dataset_directory_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            outside_image = temporary_path / "outside.png"
            outside_image.touch()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "../outside.png,anomaly,true\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "outside the dataset",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_unsupported_image_extension_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            image_path = dataset_root / "image.bmp"
            image_path.touch()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "image.bmp,normal,false\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "unsupported file extension",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_missing_image_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "missing.png,normal,false\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_duplicate_image_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            image_path = dataset_root / "image.png"
            image_path.touch()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n"
                "image.png,normal,false\n"
                "image.png,anomaly,true\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate image path",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )

    def test_empty_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            dataset_root.mkdir()

            manifest_path = temporary_path / "evaluation.csv"
            manifest_path.write_text(
                "image,group,is_anomalous\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "contains no images",
            ):
                load_labeled_image_manifest(
                    dataset_root,
                    manifest_path,
                )


if __name__ == "__main__":
    unittest.main()