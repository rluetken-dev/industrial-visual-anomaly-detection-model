# VisA Cashew q95 Transfer Validation

## Purpose

This experiment evaluates whether the normal-validation q95 threshold strategy selected provisionally during the VisA Candle calibration transfers to a previously unevaluated product category.

Cashew was evaluated without changing the established fitting, preprocessing, scoring, sampling, or threshold-quantile configuration after inspecting the Cashew test results.

The experiment validates a threshold-selection strategy, not a shared numerical threshold. Cashew derives its own numerical threshold from its own held-out normal validation images.

## Validation Question

The experiment addresses the following question:

> Does `threshold_quantile = 0.95` provide a useful initial operating point for a new category when all other established model parameters remain fixed?

## Dataset

Dataset:

```text
VisA
```

Category:

```text
cashew
```

The official one-class Cashew split contains:

| Partition | Label | Images |
|---|---|---:|
| Train | Normal | 450 |
| Test | Normal | 50 |
| Test | Anomalous | 100 |

The extracted category additionally contains 100 anomaly masks. All 600 split images and all 100 masks were confirmed to exist locally.

## Training Preparation

Only the 450 official normal training images were copied into the prepared training directory:

```text
prepared/cashew/train/normal
```

The 50 normal test images and 100 anomalous test images remained outside this directory.

The generalized exporter divided the 450 normal training images deterministically into:

| Internal partition | Images |
|---|---:|
| Feature-memory fitting | 360 |
| Threshold validation | 90 |

Split configuration:

```text
validation_fraction = 0.20
split_seed = 42
```

The generated `training_split.json` confirmed complete source coverage and the expected 360/90 partition.

## Fixed Model Configuration

The configuration was fixed before the Cashew test results were inspected:

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
| Memory chunk size | 4,096 |
| Threshold method | `normal_score_quantile` |
| Threshold quantile | 0.95 |

No parameter was adjusted in response to the Cashew evaluation.

## Exported Artifact

Artifact directory:

```text
outputs/model-artifacts/visa-cashew-generalized-q95-320
```

Artifact contents:

| File | Size |
|---|---:|
| `feature_memory.pt` | 221,185,626 bytes |
| `metadata.json` | 478 bytes |
| `training_split.json` | 7,525 bytes |

Artifact metadata:

| Property | Value |
|---|---:|
| Schema version | 2 |
| Complete feature-memory shape | 576,000 x 384 |
| Exported feature-memory shape | 144,000 x 384 |
| Exported feature-memory size | 210.94 MiB |
| Threshold | 3.134882 |

Export timings:

| Operation | Time |
|---|---:|
| Feature-memory construction | 11.88 seconds |
| Validation threshold calculation | 52.68 seconds |
| Artifact writing | 0.35 seconds |

## Evaluation Manifest

The dataset-independent evaluation manifest contains the complete official Cashew test split:

| Group | Expected label | Images |
|---|---|---:|
| Normal | Normal | 50 |
| Anomaly | Anomalous | 100 |

Manifest columns:

```csv
image,group,is_anomalous
```

All image paths are relative to the extracted VisA dataset root.

## Score Distributions

| Group | Count | Minimum | Mean | Maximum |
|---|---:|---:|---:|---:|
| Normal | 50 | 2.606198 | 2.807657 | 3.129023 |
| Anomaly | 100 | 2.775911 | 3.601031 | 5.664336 |

The highest normal test score remained slightly below the artifact threshold of `3.134882`. The normal and anomalous score distributions nevertheless overlap, so the threshold cannot separate every anomalous image from every normal image.

## Classification Results

| Metric | Result |
|---|---:|
| True positives | 66 |
| True negatives | 50 |
| False positives | 0 |
| False negatives | 34 |
| Accuracy | 0.7733 |
| Precision | 1.0000 |
| Recall | 0.6600 |
| Specificity | 1.0000 |
| F1 score | 0.7952 |

Predicted anomaly rates:

| Group | Predicted anomalous | Rate |
|---|---:|---:|
| Normal | 0 / 50 | 0.0000 |
| Anomaly | 66 / 100 | 0.6600 |

Evaluation timings:

| Operation | Time |
|---|---:|
| Artifact loading | 0.09 seconds |
| Feature-extractor creation | 0.13 seconds |
| Evaluation scoring | 90.58 seconds |

## Comparison with Candle q95

| Category | Normal test images | Anomalous test images | Precision | Recall | Specificity | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candle | 100 | 100 | 0.9324 | 0.6900 | 0.9500 | 0.7931 | 5 | 31 |
| Cashew | 50 | 100 | 1.0000 | 0.6600 | 1.0000 | 0.7952 | 0 | 34 |

The two categories achieve similar recall and F1 values despite having independently calculated numerical thresholds. Cashew produces no false positives in its official normal test partition.

## Interpretation

The fixed q95 strategy transfers plausibly from Candle to Cashew:

- recall remains close to the Candle result;
- F1 remains effectively unchanged;
- all 50 normal Cashew test images are classified correctly;
- the strategy does not require a shared absolute score threshold;
- category-specific normal validation produces the appropriate numerical threshold.

This result provides independent cross-category evidence that q95 can serve as a useful initial threshold strategy for new categories under the current pipeline.

It does not establish that q95 is universally optimal. Thirty-four of the 100 anomalous Cashew images remain undetected, and only two VisA categories have been examined with this strategy.

## Methodological Status

The q95 strategy was selected from the earlier Candle calibration before Cashew test results were inspected. The Cashew evaluation therefore serves as an independent transfer check of that fixed strategy.

The following rules apply after this evaluation:

- do not tune the Cashew quantile against the same official test split;
- do not change top fraction, memory fraction, or preprocessing based on this result and then report the same test result as independent;
- treat future Cashew parameter comparisons as exploratory calibration;
- use new evidence for any subsequent independent validation claim.

## Decision

The transfer check supports retaining the following provisional default strategy for subsequent new-category experiments:

```text
threshold_method = normal_score_quantile
threshold_quantile = 0.95
```

The numerical threshold remains category-specific and must always be calculated from that category's held-out normal validation images.

No Cashew-specific parameter optimization is authorized from this result.

## Limitations

- only 50 normal Cashew test images are available;
- 34 percent of anomalous Cashew test images are missed;
- Candle and Cashew belong to the same VisA dataset family;
- pixel-level localization quality was not measured;
- the 25-percent random feature-memory sample was not compared with complete memory for Cashew;
- the result does not represent production or regulatory validation.

## Reproducibility Notes

```text
dataset = visa
category = cashew
validation_fraction = 0.20
split_seed = 42
input_size = 320
top_fraction = 0.01
memory_fraction = 0.25
sampling_seed = 42
threshold_method = normal_score_quantile
threshold_quantile = 0.95
```

Dataset files, prepared image copies, evaluation manifests, and generated artifacts remain outside Git. This document records only aggregate results and reproducibility-relevant configuration.
