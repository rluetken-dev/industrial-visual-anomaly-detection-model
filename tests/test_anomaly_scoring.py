import unittest

import torch

from industrial_visual_anomaly_detection.models import compute_anomaly_scores

from industrial_visual_anomaly_detection.models import (
    compute_anomaly_scores,
    compute_image_scores_for_batches,
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


if __name__ == "__main__":
    unittest.main()