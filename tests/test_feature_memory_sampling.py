import unittest

import torch

from industrial_visual_anomaly_detection.models import (
    sample_feature_memory,
)


class FeatureMemorySamplingTests(unittest.TestCase):
    def test_requested_fraction_is_selected(self) -> None:
        feature_memory = torch.arange(
            40,
            dtype=torch.float32,
        ).reshape(10, 4)

        sampled_memory = sample_feature_memory(
            feature_memory,
            fraction=0.3,
            seed=42,
        )

        self.assertEqual((3, 4), tuple(sampled_memory.shape))

    def test_sampling_is_deterministic_for_equal_seed(self) -> None:
        feature_memory = torch.arange(
            80,
            dtype=torch.float32,
        ).reshape(20, 4)

        first_sample = sample_feature_memory(
            feature_memory,
            fraction=0.25,
            seed=42,
        )
        second_sample = sample_feature_memory(
            feature_memory,
            fraction=0.25,
            seed=42,
        )

        self.assertTrue(
            torch.equal(first_sample, second_sample)
        )

    def test_full_fraction_returns_complete_memory(self) -> None:
        feature_memory = torch.arange(
            12,
            dtype=torch.float32,
        ).reshape(3, 4)

        sampled_memory = sample_feature_memory(
            feature_memory,
            fraction=1.0,
        )

        self.assertTrue(
            torch.equal(feature_memory, sampled_memory)
        )

    def test_invalid_fraction_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero and at most one",
        ):
            sample_feature_memory(
                torch.zeros(10, 4),
                fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main()