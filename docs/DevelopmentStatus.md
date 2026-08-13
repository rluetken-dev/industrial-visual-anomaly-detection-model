# Industrial Visual Anomaly Detection – Development Status

## Document Purpose

This living document records the verified implementation status, experiment results, current decisions, open questions, and immediate next steps for the Industrial Visual Anomaly Detection project.

It distinguishes between implemented and tested capabilities, exploratory findings, and planned work.

## Current Phase

The project has completed its first end-to-end anomaly-detection model-development cycle and is now in the model-artifact and inference-integration phase.

The implemented pipeline can:

- validate all three locally acquired MVTec datasets;
- create deterministic fitting and validation partitions;
- preprocess images at configurable square resolutions;
- extract frozen multi-scale ResNet18 features;
- create local patch embeddings and a normal feature memory;
- compute exact chunked nearest-neighbor distances;
- produce patch-level and image-level anomaly scores;
- derive a threshold only from normal validation images;
- evaluate complete MVTec AD category test partitions;
- generate anomaly heatmaps;
- export and reload a fitted model artifact;
- classify an individual image through a command-line interface.

The current reference artifact is the MVTec AD Capsule model using a 320 × 320 input, the complete feature memory, and top-one-percent patch-score aggregation.

No .NET backend or graphical client has been implemented yet. Pixel-level localization metrics and public artifact distribution remain future work.

## Project Vision

The project is intended to become an industrial visual anomaly-detection system that:

- learns normal product appearance from defect-free images;
- detects deviations from the learned normal appearance;
- localizes suspicious regions through anomaly maps;
- evaluates detection and localization quality reproducibly;
- packages fitted model state as versioned artifacts;
- exposes inference through a client-neutral .NET backend;
- can later support desktop and web clients.

The current implementation performs unsupervised anomaly detection. It decides whether an image is normal or anomalous and produces a patch-level anomaly map. It does not classify the precise defect type.

## Verified Development Environment

| Component | Verified value |
| --- | --- |
| Operating system | Windows |
| Python | 3.12.10 |
| Virtual environment | `.venv` |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| Pillow | 12.2.0 |
| NumPy | 2.4.4 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.28.0 |
| ONNX Script | 0.7.1 |
| .NET SDK | 10.0.302 |
| Git | 2.55.0.windows.3 |

The Python version is recorded in `.python-version`, and direct dependencies are pinned in `requirements.txt`. Source compilation and `pip check` succeed. The automated test suite currently contains 54 passing tests.

## Hardware Assessment

The development system uses an AMD Radeon 860M integrated GPU. The installed PyTorch build is CPU-only, and CUDA acceleration is unavailable.

CPU compatibility is therefore a requirement. ResNet18, chunked distance calculation, configurable resolution, and optional feature-memory sampling make local development practical. Hardware acceleration may be evaluated later without becoming necessary for reproducing the baseline.

## Implemented Model Pipeline

The current PatchCore-inspired pipeline is:

```text
normal fitting images
→ deterministic preprocessing
→ frozen pretrained ResNet18
→ layer2 and layer3 feature maps
→ multi-scale patch embeddings
→ normal feature memory
→ exact chunked nearest-neighbor search
→ patch anomaly scores
→ configurable image-score aggregation
→ validation-derived threshold
→ normal/anomalous decision
```

These steps are implemented explicitly instead of being hidden behind a large anomaly-detection framework. This keeps the method understandable, testable, and suitable for later integration.

## Verified Feature Extraction

For a 224 × 224 input:

| Output | Shape |
| --- | --- |
| ResNet18 `layer2` | `(1, 128, 28, 28)` |
| ResNet18 `layer3` | `(1, 256, 14, 14)` |
| Combined feature map | `(1, 384, 28, 28)` |
| Patch embeddings | `(784, 384)` |

For a 320 × 320 input:

| Output | Shape |
| --- | --- |
| ResNet18 `layer2` | `(1, 128, 40, 40)` |
| ResNet18 `layer3` | `(1, 256, 20, 20)` |
| Patch embeddings | `(1600, 384)` |

The backbone remains in evaluation mode, and none of its parameters require gradients. The current baseline uses pretrained representations without fine-tuning.

## Verified ONNX Feasibility

The ResNet18 feature extractor was exported successfully to ONNX and passed the ONNX checker. ONNX Runtime reproduced PyTorch output with maximum absolute differences of `0.00000241` for `layer2` and `0.00000161` for `layer3`.

The provisional ONNX files prove cross-runtime feasibility but are not a complete anomaly-detection artifact. Deployment also requires preprocessing, feature memory, aggregation settings, and a threshold.

## Preprocessing Decisions

The selected preprocessing pipeline is:

```text
RGB image
→ direct resize to configured square input
→ tensor conversion
→ ImageNet normalization
```

Direct resizing was selected over TorchVision's default resize-plus-center-crop pipeline because center cropping removed part of the Bottle boundary. Bottle was evaluated at 224 × 224. The stronger Capsule baseline uses 320 × 320. Future non-square categories require a separate padding or aspect-ratio decision.

## Dataset Storage and Validation

Datasets are stored outside Git under:

```text
C:/dev/data/industrial-visual-anomaly-detection/
```

Dataset archives, extracted images, validation reports, generated model artifacts, and experiment outputs are excluded from the repository.

### MVTec AD

| Property | Verified value |
| --- | --- |
| SHA-256 | `CF4313B13603BEC67ABB49CA959488F7EEDCE2A9F7795EC54446C649AC98CD3D` |
| Categories | 15 |
| Normal training images | 3,629 |
| Normal test images | 467 |
| Anomalous test images | 1,258 |
| Ground-truth masks | 1,258 |
| Total PNG files | 6,612 |

### MVTec LOCO AD

| Property | Verified value |
| --- | --- |
| SHA-256 | `9E7C84DBA550FD2E59D8E9E231C929C45BA737B6B6A6D3814100F54D63AAE687` |
| Categories | 5 |
| Normal training images | 1,778 |
| Normal validation images | 305 |
| Normal test images | 575 |
| Logical anomaly images | 561 |
| Structural anomaly images | 432 |
| Mask groups | 993 |
| Mask files | 1,246 |
| Total PNG files | 4,897 |

MVTec LOCO AD mask values are validated against each category's `defects_config.json`.

### MVTec AD 2

| Property | Verified value |
| --- | --- |
| SHA-256 | `C0DED99EF32BFC8E352D52BEB44515E5B292B8598CB963AADFA91CA0763505E4` |
| Categories | 8 |
| Normal training images | 2,528 |
| Normal validation images | 302 |
| Public normal test images | 379 |
| Public anomalous test images | 705 |
| Public masks | 705 |
| Private test images | 2,045 |
| Private mixed test images | 2,045 |
| Total PNG files | 8,709 |

All implemented structure, readability, inventory, mask-name, and mask-content checks pass. Private MVTec AD 2 ground truth is not stored locally.

## Machine-Readable Validation Reports

Each dataset validator supports an optional `--report` argument. Schema-versioned JSON reports are written only after every validation stage succeeds. They include dataset identity, local root, inventories, image properties, mask counts, and validation-stage results. Generated reports are excluded from Git.

## Deterministic Category Splits

The category-configurable split generator uses seed `42`. Manifests use relative paths and are validated for counts, duplicates, overlap, and unsafe traversal.

| Category | Fitting | Normal validation | Manifest |
| --- | ---: | ---: | --- |
| Bottle | 167 | 42 | `configs/splits/mvtec-ad-bottle-seed-42.json` |
| Capsule | 175 | 44 | `configs/splits/mvtec-ad-capsule-seed-42.json` |

Official test images are excluded from fitting and threshold selection. Defect folders are used only for grouped evaluation.

## Feature Memory and Scoring

The complete feature memory concatenates patch embeddings from every normal fitting image. Exact nearest-neighbor distances are computed in chunks to restrict temporary memory use.

Supported image-level aggregation methods are:

- maximum patch score;
- mean of the highest configurable patch-score fraction.

Top-one-percent mean is the selected reference aggregation. The threshold is the maximum score among normal validation images, and only scores strictly above it are anomalous.

## Bottle Baseline Results

At 224 × 224 with complete feature memory:

| Aggregation | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum | 0.9398 | 1.0000 | 0.9206 | 0.9587 | 0 | 5 |
| Top 1% mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Top 5% mean | 0.9759 | 0.9692 | 1.0000 | 0.9844 | 2 | 0 |

These are exploratory results because the official test set was inspected during baseline analysis. They must not be presented as an untouched blind benchmark.

## Capsule Generalization Results

Capsule was used to test pipeline generalization beyond Bottle. Its test partition contains 23 normal and 109 anomalous images.

The selected 320 × 320, complete-memory, top-one-percent configuration produced:

| Metric | Result |
| --- | ---: |
| True positives | 104 |
| True negatives | 21 |
| False positives | 2 |
| False negatives | 5 |
| Accuracy | 0.9470 |
| Precision | 0.9811 |
| Recall | 0.9541 |
| F1 score | 0.9674 |

This configuration creates 280,000 feature-memory entries with dimension 384 and occupies approximately 410.16 MiB as float32.

## Feature-Memory Sampling Experiment

Deterministic random sampling with seed `42` was evaluated for Capsule at 320 × 320:

| Memory | Entries | Validation + test scoring | Recall | F1 | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 280,000 | 225.06 s | 0.9541 | 0.9674 | 5 |
| 75% | 210,000 | 184.33 s | 0.8991 | 0.9423 | 11 |
| 50% | 140,000 | 120.36 s | 0.8624 | 0.9216 | 15 |
| 25% | 70,000 | 64.15 s | 0.6330 | 0.7753 | 40 |

Sampling accelerates the search but degrades recall too strongly. Complete memory remains the quality baseline. Sampling remains a reproducible experimental option; a coverage-preserving coreset may be investigated later.

## Heatmap Visualization

Patch scores can be resized, colorized, and blended with source images. Both per-image normalization and fixed threshold-based normalization are supported. Threshold-based normalization is preferred for comparisons because it uses a consistent reference.

Heatmaps are explanatory outputs. Quantitative pixel-level localization evaluation remains open.

## Exported Model Artifact

Reusable artifacts are implemented as:

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

Metadata records schema version, dataset, category, backbone, input and patch-grid sizes, embedding dimension, aggregation configuration, threshold, sampling configuration, and feature-memory entry count.

The writer validates tensor shape and contents. The loader reconstructs typed metadata and loads the tensor on CPU with `weights_only=True`. Round-trip persistence and invalid configurations are covered by tests.

## Reference Capsule Artifact

| Property | Value |
| --- | --- |
| Dataset | `mvtec-ad` |
| Category | `capsule` |
| Backbone | `resnet18` |
| Input size | `320 × 320` |
| Patch grid | `40 × 40` |
| Embedding dimension | 384 |
| Feature-memory entries | 280,000 |
| Feature-memory size | 410.16 MiB |
| Aggregation | `top_fraction_mean` |
| Top fraction | 0.01 |
| Memory fraction | 1.0 |
| Threshold | 2.501821517944336 |

The artifact is stored locally below `outputs/model-artifacts/` and excluded from Git. It has not been published as a release asset.

## Single-Image Inference

The loaded artifact classifies an image without fitting or validation data. Inference recreates the configured input size from artifact metadata and applies the versioned preprocessing implementation from the Python package. It then extracts embeddings, performs exact scoring, applies the saved aggregation rule, and compares the score with the stored threshold.

The prediction contains the image path, anomaly score, threshold, Boolean decision, and patch-score map. `scripts/predict_image.py` exposes the flow through a CLI.

| Image | Score | Threshold | Decision | Prediction time |
| --- | ---: | ---: | --- | ---: |
| Capsule `test/good/000.png` | 1.848755 | 2.501822 | normal | 1.44 s |
| Capsule `test/poke/000.png` | 4.992109 | 2.501822 | anomalous | 1.46 s |

Artifact loading took approximately 0.20 seconds and extractor creation approximately 0.20 seconds. A persistent service could initialize both once.

## Automated Tests and Quality Checks

The 54 passing tests cover manifests, dataset discovery, preprocessing, patch embeddings, feature memory, sampling, nearest-neighbor distances, scoring, aggregation, metrics, visualization, and artifact persistence.

The following checks pass:

```text
unittest discovery
compileall
pip check
git diff --check
```

## Current Repository Structure

```text
configs/
  splits/
docs/
scripts/
  create_mvtec_ad_split.py
  evaluate_mvtec_ad_category.py
  export_mvtec_ad_model.py
  inspect_preprocessing.py
  predict_image.py
  validate_mvtec_ad.py
  validate_mvtec_ad_2.py
  validate_mvtec_loco_ad.py
src/
  industrial_visual_anomaly_detection/
    artifacts/
    datasets/
    models/
    evaluation.py
    inference.py
    preprocessing.py
    visualization.py
tests/
```

Generated datasets, reports, ONNX files, model artifacts, caches, and outputs are excluded by `.gitignore`.

## Architecture Direction

```text
Python model development and evaluation
→ evaluated, versioned model artifacts
→ client-neutral .NET inference backend
→ desktop and web clients
```

Python currently owns validation, fitting, evaluation, visualization, artifact export, and reference inference. The future .NET backend is expected to own persistent artifact loading, preprocessing parity, ONNX execution, scoring, stable result contracts, and multi-client access.

The current artifact uses a PyTorch tensor and a PyTorch feature extractor. A framework-neutral .NET release contract has not yet been finalized.

## Confirmed Reference Decisions

- normal-only unsupervised fitting;
- frozen pretrained ResNet18;
- `layer2` and `layer3` feature fusion;
- 384-dimensional embeddings;
- direct square resizing and ImageNet normalization;
- 320 × 320 Capsule input;
- exact chunked nearest-neighbor search;
- complete feature memory;
- top-one-percent mean aggregation;
- maximum normal-validation score as threshold;
- local artifact storage outside Git.

## Open Decisions

- preprocessing for non-square categories;
- smarter coverage-preserving coreset selection;
- approximate nearest-neighbor implementation;
- quantitative pixel-level metrics and map smoothing;
- framework-neutral feature-memory storage;
- packaging ONNX, metadata, and feature memory together;
- artifact checksums and compatibility validation;
- public release contents;
- division of inference work between ONNX and .NET;
- future threshold calibration.

## Deferred Work

- .NET inference backend;
- desktop and web clients;
- database persistence;
- authentication and authorization;
- deployment packaging;
- GPU optimization;
- backbone fine-tuning;
- multi-category artifact management;
- defect-type classification;
- private MVTec AD 2 benchmark submission;
- production monitoring and drift handling.

## Immediate Next Steps

1. Update architecture, strategy, specification, and README documents consistently.
2. Add focused tests for single-image inference and artifact compatibility.
3. Decide whether the CLI should save prediction reports and heatmaps.
4. Define a versioned, framework-neutral release layout for later .NET consumption.
5. Export and verify the neural extractor for the selected 320 × 320 configuration.
6. Evaluate pixel-level localization against Capsule masks.
7. Add machine-readable experiment summaries.
8. Design the future backend inference contract.

## Last Verified Status

As of 2026-08-13:

- all three acquired MVTec datasets are validated;
- machine-readable dataset reports are supported;
- Bottle and Capsule deterministic manifests are versioned;
- the reusable Python anomaly pipeline is implemented;
- configurable 224 × 224 and 320 × 320 preprocessing is verified;
- exact patch-level anomaly scoring and configurable aggregation are implemented;
- Bottle and Capsule evaluations have been completed;
- the selected Capsule baseline achieves 0.9541 recall and 0.9674 F1 in the explored evaluation;
- random memory sampling was evaluated and rejected as the quality baseline;
- anomaly heatmaps are implemented;
- artifacts can be exported and loaded;
- a 410.16 MiB Capsule artifact was verified;
- normal and anomalous Capsule images were classified correctly through the CLI;
- observed single-image prediction time is approximately 1.45 seconds on CPU;
- 54 automated tests pass;
- no .NET backend, graphical client, or public artifact release exists yet.

The next active milestone is documentation consolidation followed by definition of a stable cross-runtime inference contract.
