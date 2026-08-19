import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.datasets.image_split import (
    create_image_path_split,
)


class ImagePathSplitTests(unittest.TestCase):
    def test_all_paths_are_split_without_overlap(self) -> None:
        image_paths = tuple(
            Path(f"image-{index:02d}.png")
            for index in range(10)
        )

        split = create_image_path_split(
            image_paths,
            validation_fraction=0.2,
            seed=42,
        )

        self.assertEqual(8, len(split.fitting_paths))
        self.assertEqual(2, len(split.validation_paths))
        self.assertEqual(
            set(image_paths),
            set(split.fitting_paths) | set(split.validation_paths),
        )
        self.assertFalse(
            set(split.fitting_paths) & set(split.validation_paths)
        )

    def test_equal_inputs_and_seed_produce_equal_splits(self) -> None:
        image_paths = tuple(
            Path(f"image-{index:02d}.png")
            for index in range(10)
        )

        first_split = create_image_path_split(image_paths, seed=42)
        second_split = create_image_path_split(image_paths, seed=42)

        self.assertEqual(first_split, second_split)

    def test_different_seeds_can_produce_different_splits(self) -> None:
        image_paths = tuple(
            Path(f"image-{index:02d}.png")
            for index in range(10)
        )

        first_split = create_image_path_split(image_paths, seed=42)
        second_split = create_image_path_split(image_paths, seed=43)

        self.assertNotEqual(
            first_split.validation_paths,
            second_split.validation_paths,
        )

    def test_small_dataset_keeps_both_partitions_non_empty(self) -> None:
        image_paths = (
            Path("first.png"),
            Path("second.png"),
        )

        split = create_image_path_split(image_paths)

        self.assertEqual(1, len(split.fitting_paths))
        self.assertEqual(1, len(split.validation_paths))

    def test_fewer_than_two_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires at least two image paths",
        ):
            create_image_path_split((Path("only.png"),))

    def test_duplicate_paths_are_rejected(self) -> None:
        duplicate_path = Path("duplicate.png")

        with self.assertRaisesRegex(
            ValueError,
            "must not contain duplicates",
        ):
            create_image_path_split(
                (duplicate_path, duplicate_path)
            )

    def test_invalid_validation_fractions_are_rejected(self) -> None:
        image_paths = (
            Path("first.png"),
            Path("second.png"),
        )

        for validation_fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(
                validation_fraction=validation_fraction
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Validation fraction",
                ):
                    create_image_path_split(
                        image_paths,
                        validation_fraction=validation_fraction,
                    )


if __name__ == "__main__":
    unittest.main()