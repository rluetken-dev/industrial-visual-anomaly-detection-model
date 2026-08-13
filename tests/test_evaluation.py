import unittest

import torch

from industrial_visual_anomaly_detection.evaluation import (
    classify_anomaly_scores,
    compute_binary_classification_metrics,
    select_maximum_normal_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_maximum_normal_score_becomes_threshold(self) -> None:
        threshold = select_maximum_normal_threshold(
            torch.tensor([2.0, 4.5, 3.0])
        )

        self.assertAlmostEqual(4.5, threshold)

    def test_only_scores_above_threshold_are_anomalous(self) -> None:
        predictions = classify_anomaly_scores(
            torch.tensor([3.9, 4.0, 4.1]),
            threshold=4.0,
        )

        self.assertTrue(
            torch.equal(
                predictions,
                torch.tensor([False, False, True]),
            )
        )

    def test_binary_metrics_are_computed_correctly(self) -> None:
        predictions = torch.tensor(
            [True, True, False, False, True]
        )
        expected_labels = torch.tensor(
            [True, False, True, False, True]
        )

        metrics = compute_binary_classification_metrics(
            predictions,
            expected_labels,
        )

        self.assertEqual(2, metrics.true_positives)
        self.assertEqual(1, metrics.true_negatives)
        self.assertEqual(1, metrics.false_positives)
        self.assertEqual(1, metrics.false_negatives)
        self.assertAlmostEqual(0.6, metrics.accuracy)
        self.assertAlmostEqual(2 / 3, metrics.precision)
        self.assertAlmostEqual(2 / 3, metrics.recall)
        self.assertAlmostEqual(2 / 3, metrics.f1_score)

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            compute_binary_classification_metrics(
                torch.tensor([True, False]),
                torch.tensor([True]),
            )


if __name__ == "__main__":
    unittest.main()