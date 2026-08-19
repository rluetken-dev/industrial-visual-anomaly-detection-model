# VisA Candle Threshold Calibration

## Purpose

This experiment evaluates how the normal-validation score quantile affects image-level anomaly classification for the VisA Candle category.

The experiment compares three artifacts that use identical training data, feature memory, preprocessing, model architecture, and score aggregation. Only the threshold quantile differs.

## Dataset

Dataset:

```text
VisA
```

Category:

```text
candle
```

The official one-class split contains:

| Partition | Label | Images |
|---|---|---:|
| Train | Normal | 900 |
| Test | Normal | 100 |
| Test | Anomalous | 100 |

The 900 normal training images were divided deterministically into:

| Internal partition | Images |
|---|---:|
| Feature-memory fitting | 720 |
| Threshold validation | 180 |

Split seed:

```text
42
```

Validation fraction:

```text
0.20
```

The official 200 test images were described through a dataset-independent evaluation manifest.

## Shared Model Configuration

All compared artifacts used the following configuration:

| Setting | Value |
|---|---|
| Backbone | ResNet18 |
| Input size | 320 x 320 |
| Patch grid | 40 x 40 |
| Embedding dimension | 384 |
| Aggregation method | `top_fraction_mean` |
| Top fraction | 0.01 |
| Memory fraction | 0.25 |
| Sampling seed | 42 |
| Feature-memory entries | 288,000 |

The feature-memory files of the q100, q99, and q95 artifacts had identical SHA-256 hashes. This confirms that the observed classification changes were caused by threshold calibration rather than different feature memories.

## Compared Thresholds

Three normal-validation score quantiles were evaluated:

| Variant | Quantile | Threshold |
|---|---:|---:|
| q100 | 1.00 | 3.373678 |
| q99 | 0.99 | 3.001366 |
| q95 | 0.95 | 2.763051 |

A lower quantile lowers the anomaly threshold. This generally increases anomaly recall while also increasing the risk of false-positive decisions.

## Test Score Distributions

The image-level score distributions remained identical for every threshold variant:

| Group | Count | Minimum | Mean | Maximum |
|---|---:|---:|---:|---:|
| Normal | 100 | 1.838488 | 2.267601 | 2.967893 |
| Anomalous | 100 | 2.292779 | 3.126182 | 5.035840 |

The distributions overlap. Therefore, no threshold can perfectly separate every normal and anomalous image in this evaluation set.

## Classification Results

| Variant | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q100 | 31 | 100 | 0 | 69 | 0.6550 | 1.0000 | 0.3100 | 1.0000 | 0.4733 |
| q99 | 56 | 100 | 0 | 44 | 0.7800 | 1.0000 | 0.5600 | 1.0000 | 0.7179 |
| q95 | 69 | 95 | 5 | 31 | 0.8200 | 0.9324 | 0.6900 | 0.9500 | 0.7931 |

## Interpretation

The maximum-normal threshold represented by q100 is too conservative for this category. It avoids false positives but detects only 31 percent of the anomalous test images.

The q99 threshold improves recall to 56 percent without producing false positives in this test set.

The q95 threshold provides the strongest measured balance:

- recall increases to 69 percent;
- false negatives decrease to 31;
- five normal images are classified as anomalous;
- specificity remains at 95 percent;
- the F1 score increases to 0.7931.

For an inspection workflow in which anomalous decisions trigger human review, q95 is the preferred provisional calibration candidate. In that context, a moderate number of additional inspections may be preferable to missing more true anomalies.

## Methodological Limitation

The official VisA Candle test images were used to compare the three threshold variants.

Consequently, q95 must not be presented as independently validated solely on this test set. Selecting a quantile after observing these results introduces test-set feedback into the calibration decision.

The current result is therefore an exploratory calibration result rather than a final unbiased performance estimate.

## Decision

Use the following threshold setting as the provisional calibration candidate for subsequent validation:

```text
threshold_method = normal_score_quantile
threshold_quantile = 0.95
```

Do not continue tuning the quantile against the same Candle test images.

The next validation step should keep this threshold strategy fixed and evaluate it on previously unused data or another suitable category.

## Reproducibility Notes

The experiment used deterministic image splitting and feature-memory sampling:

```text
split_seed = 42
sampling_seed = 42
```

The q95 artifact was exported to:

```text
outputs/model-artifacts/visa-candle-generalized-q95-320
```

Generated artifacts and dataset files remain outside Git.
