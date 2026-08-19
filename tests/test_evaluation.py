import unittest

import torch

from industrial_visual_anomaly_detection.evaluation import (
    classify_anomaly_scores,
    compute_binary_classification_metrics,
    select_maximum_normal_threshold,
    select_normal_score_quantile_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_maximum_normal_score_becomes_threshold(self) -> None:
        threshold = select_maximum_normal_threshold(
            torch.tensor([2.0, 4.5, 3.0])
        )

        self.assertAlmostEqual(4.5, threshold)

    def test_normal_score_quantile_becomes_threshold(
        self,
    ) -> None:
        threshold = select_normal_score_quantile_threshold(
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            quantile=0.75,
        )

        self.assertAlmostEqual(3.25, threshold)

    def test_quantile_one_matches_maximum_threshold(
        self,
    ) -> None:
        scores = torch.tensor([2.0, 4.5, 3.0])

        quantile_threshold = (
            select_normal_score_quantile_threshold(
                scores,
                quantile=1.0,
            )
        )
        maximum_threshold = (
            select_maximum_normal_threshold(scores)
        )

        self.assertAlmostEqual(
            maximum_threshold,
            quantile_threshold,
        )

    def test_invalid_threshold_quantile_is_rejected(
        self,
    ) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0])

        for quantile in (0.0, -0.1, 1.1):
            with self.subTest(quantile=quantile):
                with self.assertRaisesRegex(
                    ValueError,
                    "greater than zero and at most one",
                ):
                    select_normal_score_quantile_threshold(
                        scores,
                        quantile=quantile,
                    )

    def test_empty_quantile_scores_are_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires normal validation scores",
        ):
            select_normal_score_quantile_threshold(
                torch.tensor([]),
                quantile=0.95,
            )

    def test_non_finite_quantile_scores_are_rejected(
        self,
    ) -> None:
        for invalid_score in (
            float("nan"),
            float("inf"),
            float("-inf"),
        ):
            with self.subTest(invalid_score=invalid_score):
                with self.assertRaisesRegex(
                    ValueError,
                    "only finite values",
                ):
                    select_normal_score_quantile_threshold(
                        torch.tensor(
                            [1.0, invalid_score]
                        ),
                        quantile=0.95,
                    )

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