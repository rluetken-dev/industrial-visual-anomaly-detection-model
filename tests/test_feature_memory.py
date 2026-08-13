import unittest

import torch

from industrial_visual_anomaly_detection.models import build_feature_memory


class FakeEmbeddingExtractor(torch.nn.Module):
    """Create two simple three-dimensional embeddings per input image."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_values = images[:, 0, 0, 0].reshape(-1, 1)

        first_patch = torch.cat(
            [image_values, image_values + 1, image_values + 2],
            dim=1,
        )
        second_patch = first_patch + 10

        return torch.stack(
            [first_patch, second_patch],
            dim=1,
        ).reshape(-1, 3)


class FeatureMemoryTests(unittest.TestCase):
    def test_feature_memory_combines_embeddings_from_all_batches(self) -> None:
        first_batch = torch.tensor(
            [
                [[[1.0]]],
                [[[2.0]]],
            ]
        )
        second_batch = torch.tensor(
            [
                [[[3.0]]],
            ]
        )

        batches = [
            (first_batch, ["first.png", "second.png"]),
            (second_batch, ["third.png"]),
        ]

        memory = build_feature_memory(
            batches,
            FakeEmbeddingExtractor(),
        )

        self.assertEqual((6, 3), tuple(memory.shape))
        self.assertFalse(memory.requires_grad)
        self.assertTrue(torch.isfinite(memory).all().item())

        expected = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [11.0, 12.0, 13.0],
                [2.0, 3.0, 4.0],
                [12.0, 13.0, 14.0],
                [3.0, 4.0, 5.0],
                [13.0, 14.0, 15.0],
            ]
        )

        self.assertTrue(torch.equal(expected, memory))

    def test_empty_batches_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "at least one image batch",
        ):
            build_feature_memory(
                [],
                FakeEmbeddingExtractor(),
            )


if __name__ == "__main__":
    unittest.main()