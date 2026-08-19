import json
import tempfile
import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets.image_split import (
    ImagePathSplit,
)
from industrial_visual_anomaly_detection.datasets.image_split_manifest import (
    save_image_path_split_manifest,
)
from industrial_visual_anomaly_detection.datasets.split_manifest_loader import (
    load_split_manifest,
)


class ImagePathSplitManifestTests(unittest.TestCase):
    def test_split_manifest_is_saved_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            nested_directory = image_directory / "nested"
            nested_directory.mkdir(parents=True)

            fitting_path = image_directory / "fitting.png"
            validation_path = (
                nested_directory / "validation.jpg"
            )
            fitting_path.touch()
            validation_path.touch()

            split = ImagePathSplit(
                fitting_paths=(fitting_path,),
                validation_paths=(validation_path,),
            )
            output_path = root / "artifact" / "training_split.json"

            saved_path = save_image_path_split_manifest(
                image_directory=image_directory,
                split=split,
                dataset="custom-dataset",
                category="bottle",
                validation_fraction=0.5,
                seed=42,
                output_path=output_path,
            )

            self.assertEqual(output_path.resolve(), saved_path)

            manifest_data = json.loads(
                saved_path.read_text(encoding="utf-8")
            )

            self.assertEqual(
                ["fitting.png"],
                manifest_data["fitting_images"],
            )
            self.assertEqual(
                ["nested/validation.jpg"],
                manifest_data["validation_images"],
            )
            self.assertEqual(".", manifest_data["source_directory"])
            self.assertEqual(2, manifest_data["source_image_count"])

    def test_saved_manifest_is_compatible_with_manifest_loader(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            image_directory.mkdir()

            fitting_path = image_directory / "fitting.png"
            validation_path = image_directory / "validation.png"
            fitting_path.touch()
            validation_path.touch()

            output_path = root / "training_split.json"

            save_image_path_split_manifest(
                image_directory=image_directory,
                split=ImagePathSplit(
                    fitting_paths=(fitting_path,),
                    validation_paths=(validation_path,),
                ),
                dataset="custom-dataset",
                category="bottle",
                validation_fraction=0.5,
                seed=42,
                output_path=output_path,
            )

            manifest = load_split_manifest(output_path)

            self.assertEqual("custom-dataset", manifest.dataset)
            self.assertEqual("bottle", manifest.category)
            self.assertEqual(42, manifest.seed)
            self.assertEqual(
                (Path("fitting.png"),),
                manifest.fitting_images,
            )
            self.assertEqual(
                (Path("validation.png"),),
                manifest.validation_images,
            )

    def test_image_outside_source_directory_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            image_directory.mkdir()

            fitting_path = image_directory / "fitting.png"
            outside_path = root / "outside.png"
            fitting_path.touch()
            outside_path.touch()

            with self.assertRaisesRegex(
                ValueError,
                "outside the image directory",
            ):
                save_image_path_split_manifest(
                    image_directory=image_directory,
                    split=ImagePathSplit(
                        fitting_paths=(fitting_path,),
                        validation_paths=(outside_path,),
                    ),
                    dataset="custom-dataset",
                    category="bottle",
                    validation_fraction=0.5,
                    seed=42,
                    output_path=root / "training_split.json",
                )


if __name__ == "__main__":
    unittest.main()