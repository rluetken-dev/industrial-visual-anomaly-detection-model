import unittest

import torch

from industrial_visual_anomaly_detection.models import (
    aggregate_patch_scores,
    aggregate_top_patch_scores,
    compute_anomaly_scores,
    compute_image_scores_for_batches,
    compute_patch_scores_for_batches,
)


class FakeEmbeddingExtractor(torch.nn.Module):
    """Create 784 identical one-dimensional patch embeddings per image."""

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        image_values = image_batch.reshape(image_batch.shape[0], 1)

        return image_values.repeat_interleave(
            28 * 28,
            dim=0,
        )


class AnomalyScoringTests(unittest.TestCase):
    def test_batch_scoring_preserves_image_and_path_order(self) -> None:
        batches = [
            (
                torch.tensor([[1.0], [2.0]]),
                ["normal-001.png", "normal-002.png"],
            ),
            (
                torch.tensor([[3.0]]),
                ["normal-003.png"],
            ),
        ]

        image_scores, image_paths = compute_image_scores_for_batches(
            batches,
            FakeEmbeddingExtractor(),
            torch.tensor([[0.0]]),
            memory_chunk_size=1,
        )

        self.assertTrue(
            torch.allclose(
                image_scores,
                torch.tensor([1.0, 2.0, 3.0]),
                atol=1e-6,
            )
        )
        self.assertEqual(
            (
                "normal-001.png",
                "normal-002.png",
                "normal-003.png",
            ),
            image_paths,
        )

    def test_empty_batch_collection_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "at least one batch",
        ):
            compute_image_scores_for_batches(
                [],
                FakeEmbeddingExtractor(),
                torch.tensor([[0.0]]),
            )

    def test_patch_and_image_score_shapes_are_correct(self) -> None:
        feature_memory = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 10.0],
            ]
        )
        query_embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
            ]
        )

        result = compute_anomaly_scores(
            query_embeddings,
            feature_memory,
            patch_grid_size=(2, 2),
            memory_chunk_size=1,
        )

        self.assertEqual((1, 2, 2), tuple(result.patch_scores.shape))
        self.assertEqual((1,), tuple(result.image_scores.shape))

    def test_maximum_patch_distance_becomes_image_score(self) -> None:
        feature_memory = torch.tensor(
            [
                [0.0, 0.0],
            ]
        )
        query_embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
            ]
        )

        result = compute_anomaly_scores(
            query_embeddings,
            feature_memory,
            patch_grid_size=(2, 2),
        )

        expected_patch_scores = torch.tensor(
            [
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                ]
            ]
        )

        self.assertTrue(
            torch.allclose(
                result.patch_scores,
                expected_patch_scores,
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(
            3.0,
            result.image_scores[0].item(),
            places=6,
        )

    def test_multiple_images_are_scored_separately(self) -> None:
        feature_memory = torch.tensor(
            [
                [0.0],
            ]
        )
        query_embeddings = torch.tensor(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [6.0],
                [7.0],
                [8.0],
            ]
        )

        result = compute_anomaly_scores(
            query_embeddings,
            feature_memory,
            patch_grid_size=(2, 2),
        )

        self.assertEqual((2, 2, 2), tuple(result.patch_scores.shape))
        self.assertTrue(
            torch.equal(
                result.image_scores,
                torch.tensor([4.0, 8.0]),
            )
        )

    def test_invalid_embedding_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "divisible",
        ):
            compute_anomaly_scores(
                torch.zeros(5, 2),
                torch.zeros(3, 2),
                patch_grid_size=(2, 2),
            )

    def test_top_patch_fraction_is_averaged_per_image(self) -> None:
        patch_scores = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
            ]
        )

        image_scores = aggregate_top_patch_scores(
            patch_scores,
            top_fraction=0.5,
        )

        self.assertTrue(
            torch.allclose(
                image_scores,
                torch.tensor([3.5, 35.0]),
            )
        )

    def test_small_top_fraction_selects_at_least_one_patch(self) -> None:
        image_scores = aggregate_top_patch_scores(
            torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                ]
            ),
            top_fraction=0.01,
        )

        self.assertTrue(
            torch.equal(
                image_scores,
                torch.tensor([4.0]),
            )
        )

    def test_invalid_top_fraction_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero and at most one",
        ):
            aggregate_top_patch_scores(
                torch.zeros(1, 2, 2),
                top_fraction=0.0,
            )

    def test_patch_batch_scoring_preserves_grids_and_paths(self) -> None:
        batches = [
            (
                torch.tensor([[1.0], [2.0]]),
                ["image-001.png", "image-002.png"],
            )
        ]

        patch_scores, image_paths = compute_patch_scores_for_batches(
            batches,
            FakeEmbeddingExtractor(),
            torch.tensor([[0.0]]),
            memory_chunk_size=1,
        )

        self.assertEqual((2, 28, 28), tuple(patch_scores.shape))
        self.assertEqual(
            ("image-001.png", "image-002.png"),
            image_paths,
        )
        self.assertTrue(
            torch.equal(
                patch_scores[:, 0, 0],
                torch.tensor([1.0, 2.0]),
            )
        )

    def test_maximum_aggregation_selects_highest_patch(self) -> None:
        image_scores = aggregate_patch_scores(
            torch.tensor(
                [
                    [
                        [1.0, 4.0],
                        [2.0, 3.0],
                    ]
                ]
            ),
            method="maximum",
        )

        self.assertTrue(
            torch.equal(
                image_scores,
                torch.tensor([4.0]),
            )
        )

    def test_unknown_aggregation_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported",
        ):
            aggregate_patch_scores(
                torch.zeros(1, 2, 2),
                method="unknown",
            )

    def test_anomaly_scoring_supports_top_fraction_mean(self) -> None:
        result = compute_anomaly_scores(
            query_embeddings=torch.tensor(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                ]
            ),
            feature_memory=torch.tensor([[0.0]]),
            patch_grid_size=(2, 2),
            aggregation_method="top_fraction_mean",
            top_fraction=0.5,
        )

        self.assertAlmostEqual(
            3.5,
            result.image_scores[0].item(),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
