from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Contain binary anomaly-classification results."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def select_maximum_normal_threshold(
    normal_validation_scores: torch.Tensor,
) -> float:
    """Select the highest normal validation score as threshold."""

    if normal_validation_scores.numel() == 0:
        raise ValueError(
            "Threshold selection requires normal validation scores."
        )

    if not torch.isfinite(normal_validation_scores).all():
        raise ValueError(
            "Normal validation scores must contain only finite values."
        )

    return normal_validation_scores.max().item()


def select_normal_score_quantile_threshold(
    normal_validation_scores: torch.Tensor,
    quantile: float,
) -> float:
    """Select a normal-validation score quantile as threshold."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError(
            "Threshold quantile must be greater than zero "
            "and at most one."
        )

    if normal_validation_scores.numel() == 0:
        raise ValueError(
            "Threshold selection requires normal validation scores."
        )

    if not torch.isfinite(normal_validation_scores).all():
        raise ValueError(
            "Normal validation scores must contain only finite values."
        )

    quantile_scores = normal_validation_scores

    if not quantile_scores.is_floating_point():
        quantile_scores = quantile_scores.to(
            dtype=torch.float64
        )

    return torch.quantile(
        quantile_scores,
        quantile,
    ).item()


def classify_anomaly_scores(
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Classify scores strictly above the threshold as anomalous."""

    if scores.numel() == 0:
        raise ValueError("Classification requires at least one score.")

    if not torch.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")

    return scores > threshold


def compute_binary_classification_metrics(
    predictions: torch.Tensor,
    expected_labels: torch.Tensor,
) -> BinaryClassificationMetrics:
    """Compute binary classification metrics for anomaly predictions."""

    if predictions.numel() == 0:
        raise ValueError("Metric computation requires predictions.")

    if predictions.shape != expected_labels.shape:
        raise ValueError(
            "Predictions and expected labels must have equal shapes."
        )

    predicted_anomalies = predictions.bool()
    expected_anomalies = expected_labels.bool()

    true_positives = int(
        (predicted_anomalies & expected_anomalies).sum().item()
    )
    true_negatives = int(
        (~predicted_anomalies & ~expected_anomalies).sum().item()
    )
    false_positives = int(
        (predicted_anomalies & ~expected_anomalies).sum().item()
    )
    false_negatives = int(
        (~predicted_anomalies & expected_anomalies).sum().item()
    )

    sample_count = predictions.numel()
    predicted_positive_count = true_positives + false_positives
    actual_positive_count = true_positives + false_negatives

    accuracy = (
        true_positives + true_negatives
    ) / sample_count

    precision = (
        true_positives / predicted_positive_count
        if predicted_positive_count > 0
        else 0.0
    )
    recall = (
        true_positives / actual_positive_count
        if actual_positive_count > 0
        else 0.0
    )
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return BinaryClassificationMetrics(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )