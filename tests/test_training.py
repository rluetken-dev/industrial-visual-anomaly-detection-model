import unittest

from industrial_visual_anomaly_detection.training import (
    ModelTrainingConfiguration,
    train_model_artifact,
)

from pathlib import Path


class ModelTrainingConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        configuration = ModelTrainingConfiguration()

        self.assertEqual(8, configuration.batch_size)
        self.assertEqual(4096, configuration.memory_chunk_size)
        self.assertEqual(224, configuration.input_size)
        self.assertEqual(0.01, configuration.top_fraction)
        self.assertEqual(1.0, configuration.memory_fraction)
        self.assertEqual(42, configuration.sampling_seed)

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Batch size must be greater than zero",
        ):
            ModelTrainingConfiguration(batch_size=0)

    def test_non_positive_memory_chunk_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Memory chunk size must be greater than zero",
        ):
            ModelTrainingConfiguration(memory_chunk_size=0)

    def test_invalid_input_sizes_are_rejected(self) -> None:
        for input_size in (0, -32, 225):
            with self.subTest(input_size=input_size):
                with self.assertRaises(ValueError):
                    ModelTrainingConfiguration(input_size=input_size)

    def test_invalid_top_fractions_are_rejected(self) -> None:
        for top_fraction in (0.0, -0.1, 1.1):
            with self.subTest(top_fraction=top_fraction):
                with self.assertRaisesRegex(
                    ValueError,
                    "Top fraction",
                ):
                    ModelTrainingConfiguration(
                        top_fraction=top_fraction
                    )

    def test_invalid_memory_fractions_are_rejected(self) -> None:
        for memory_fraction in (0.0, -0.1, 1.1):
            with self.subTest(memory_fraction=memory_fraction):
                with self.assertRaisesRegex(
                    ValueError,
                    "Memory fraction",
                ):
                    ModelTrainingConfiguration(
                        memory_fraction=memory_fraction
                    )

    def test_missing_fitting_images_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Fitting images are required",
        ):
            train_model_artifact(
                fitting_paths=(),
                validation_paths=(Path("validation.png"),),
                dataset="custom",
                category="bottle",
                output_directory=Path("output"),
                configuration=ModelTrainingConfiguration(),
            )

    def test_missing_validation_images_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Validation images are required",
        ):
            train_model_artifact(
                fitting_paths=(Path("fitting.png"),),
                validation_paths=(),
                dataset="custom",
                category="bottle",
                output_directory=Path("output"),
                configuration=ModelTrainingConfiguration(),
            )

    def test_overlapping_image_partitions_are_rejected(self) -> None:
        shared_path = Path("shared.png")

        with self.assertRaisesRegex(
            ValueError,
            "must not overlap",
        ):
            train_model_artifact(
                fitting_paths=(shared_path,),
                validation_paths=(shared_path,),
                dataset="custom",
                category="bottle",
                output_directory=Path("output"),
                configuration=ModelTrainingConfiguration(),
            )

    def test_missing_dataset_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Dataset name is required",
        ):
            train_model_artifact(
                fitting_paths=(Path("fitting.png"),),
                validation_paths=(Path("validation.png"),),
                dataset=" ",
                category="bottle",
                output_directory=Path("output"),
                configuration=ModelTrainingConfiguration(),
            )

    def test_missing_category_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Category name is required",
        ):
            train_model_artifact(
                fitting_paths=(Path("fitting.png"),),
                validation_paths=(Path("validation.png"),),
                dataset="custom",
                category=" ",
                output_directory=Path("output"),
                configuration=ModelTrainingConfiguration(),
            )

if __name__ == "__main__":
    unittest.main()