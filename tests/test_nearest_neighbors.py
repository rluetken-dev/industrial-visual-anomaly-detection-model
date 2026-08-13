import unittest

import torch

from industrial_visual_anomaly_detection.models import (
    compute_nearest_neighbor_distances,
)


class NearestNeighborDistanceTests(unittest.TestCase):
    def test_exact_nearest_distances_are_computed(self) -> None:
        feature_memory = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ]
        )
        query_embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )

        distances = compute_nearest_neighbor_distances(
            query_embeddings,
            feature_memory,
        )

        expected = torch.tensor([0.0, 1.0, 2.0])

        self.assertTrue(
            torch.allclose(
                distances,
                expected,
                atol=1e-6,
            )
        )

    def test_chunking_produces_the_same_result(self) -> None:
        torch.manual_seed(42)

        feature_memory = torch.rand(11, 4)
        query_embeddings = torch.rand(5, 4)

        unchunked = compute_nearest_neighbor_distances(
            query_embeddings,
            feature_memory,
            memory_chunk_size=11,
        )
        chunked = compute_nearest_neighbor_distances(
            query_embeddings,
            feature_memory,
            memory_chunk_size=3,
        )

        self.assertTrue(
            torch.allclose(
                chunked,
                unchunked,
                atol=1e-6,
            )
        )

    def test_feature_dimension_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "same feature dimension",
        ):
            compute_nearest_neighbor_distances(
                torch.zeros(2, 3),
                torch.zeros(4, 5),
            )

    def test_empty_feature_memory_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "at least one embedding",
        ):
            compute_nearest_neighbor_distances(
                torch.zeros(2, 3),
                torch.empty(0, 3),
            )


if __name__ == "__main__":
    unittest.main()