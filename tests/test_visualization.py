import unittest

import torch

from industrial_visual_anomaly_detection.visualization import (
    colorize_anomaly_map,
    create_heatmap_overlay,
    normalize_anomaly_map,
    resize_anomaly_map,
    normalize_anomaly_map_by_threshold,
)


class VisualizationTests(unittest.TestCase):
    def test_anomaly_map_is_normalized_to_zero_and_one(self) -> None:
        anomaly_map = torch.tensor(
            [
                [2.0, 4.0],
                [6.0, 10.0],
            ]
        )

        normalized_map = normalize_anomaly_map(anomaly_map)

        self.assertAlmostEqual(0.0, normalized_map.min().item())
        self.assertAlmostEqual(1.0, normalized_map.max().item())

    def test_constant_map_becomes_zero(self) -> None:
        normalized_map = normalize_anomaly_map(
            torch.full((2, 3), 5.0)
        )

        self.assertTrue(
            torch.equal(
                normalized_map,
                torch.zeros(2, 3),
            )
        )

    def test_anomaly_map_is_resized_to_requested_shape(self) -> None:
        resized_map = resize_anomaly_map(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            output_size=(8, 12),
        )

        self.assertEqual((8, 12), tuple(resized_map.shape))

    def test_invalid_output_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            resize_anomaly_map(
                torch.zeros(2, 2),
                output_size=(0, 10),
            )

    def test_normalized_map_is_converted_to_rgb_heatmap(self) -> None:
        heatmap = colorize_anomaly_map(
            torch.tensor(
                [
                    [0.0, 0.5, 1.0],
                ]
            )
        )

        self.assertEqual((3, 1, 3), tuple(heatmap.shape))

        self.assertTrue(
            torch.allclose(
                heatmap[:, 0, 0],
                torch.tensor([0.0, 0.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                heatmap[:, 0, 1],
                torch.tensor([0.5, 1.0, 0.5]),
            )
        )
        self.assertTrue(
            torch.allclose(
                heatmap[:, 0, 2],
                torch.tensor([1.0, 0.0, 0.0]),
            )
        )

    def test_out_of_range_normalized_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "between zero and one",
        ):
            colorize_anomaly_map(
                torch.tensor(
                    [
                        [-0.1, 1.1],
                    ]
                )
            )

    def test_image_and_heatmap_are_blended(self) -> None:
        image = torch.zeros(3, 2, 2)
        heatmap = torch.ones(3, 2, 2)

        overlay = create_heatmap_overlay(
            image,
            heatmap,
            opacity=0.25,
        )

        self.assertTrue(
            torch.allclose(
                overlay,
                torch.full((3, 2, 2), 0.25),
            )
        )

    def test_overlay_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "equal shapes",
        ):
            create_heatmap_overlay(
                torch.zeros(3, 4, 4),
                torch.zeros(3, 2, 2),
            )

    def test_invalid_opacity_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "between zero and one",
        ):
            create_heatmap_overlay(
                torch.zeros(3, 2, 2),
                torch.zeros(3, 2, 2),
                opacity=1.5,
            )

    def test_anomaly_map_is_normalized_against_threshold(self) -> None:
        normalized_map = normalize_anomaly_map_by_threshold(
            torch.tensor(
                [
                    [0.0, 2.0, 4.0, 8.0],
                ]
            ),
            threshold=4.0,
        )

        self.assertTrue(
            torch.allclose(
                normalized_map,
                torch.tensor(
                    [
                        [0.0, 0.5, 1.0, 1.0],
                    ]
                ),
            )
        )

    def test_non_positive_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            normalize_anomaly_map_by_threshold(
                torch.zeros(2, 2),
                threshold=0.0,
            )


if __name__ == "__main__":
    unittest.main()