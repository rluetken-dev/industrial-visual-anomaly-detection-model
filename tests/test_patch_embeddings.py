import unittest

import torch

from industrial_visual_anomaly_detection.models import create_patch_embeddings


class PatchEmbeddingTests(unittest.TestCase):
    def test_single_image_produces_expected_shape(self) -> None:
        layer2 = torch.ones(1, 128, 28, 28)
        layer3 = torch.full((1, 256, 14, 14), 2.0)

        embeddings = create_patch_embeddings(layer2, layer3)

        self.assertEqual((784, 384), tuple(embeddings.shape))
        self.assertTrue(torch.isfinite(embeddings).all().item())

    def test_feature_channels_are_concatenated_in_expected_order(self) -> None:
        layer2 = torch.ones(1, 128, 28, 28)
        layer3 = torch.full((1, 256, 14, 14), 2.0)

        embeddings = create_patch_embeddings(layer2, layer3)
        first_embedding = embeddings[0]

        self.assertTrue(
            torch.equal(
                first_embedding[:128],
                torch.ones(128),
            )
        )
        self.assertTrue(
            torch.equal(
                first_embedding[128:],
                torch.full((256,), 2.0),
            )
        )

    def test_batch_embeddings_are_flattened_by_image_and_position(self) -> None:
        layer2 = torch.ones(2, 128, 28, 28)
        layer3 = torch.ones(2, 256, 14, 14)

        embeddings = create_patch_embeddings(layer2, layer3)

        self.assertEqual((1568, 384), tuple(embeddings.shape))


if __name__ == "__main__":
    unittest.main()