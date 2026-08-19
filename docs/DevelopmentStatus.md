# Industrial Visual Anomaly Detection – Development Status

## Document Purpose

This living document records verified implementation progress, experiment results, current decisions, open questions, and immediate next steps for the Industrial Visual Anomaly Detection model project.

It distinguishes between automated verification, manually verified integration behavior, exploratory benchmark findings, and planned work.

## Current Phase

The project has completed its initial model-development, artifact-export, inference-service, backend-integration, WPF desktop-client, heatmap-visualization, and Docker Compose orchestration milestones.

The current phase generalizes category-specific model fitting so that an artifact can be created directly from an external directory of normal PNG or JPEG images. The first generalized directory-based export was verified with MVTec AD Bottle images. Multi-artifact runtime selection and evaluation on a genuinely non-MVTec image collection remain future work.

The implemented system can:

- validate all three locally acquired MVTec datasets;
- discover normal PNG and JPEG images recursively in external directories;
- create deterministic fitting and validation partitions;
- record generated partitions with portable relative paths in `training_split.json`;
- extract frozen multi-scale ResNet18 features;
- build category-specific normal feature memories;
- calculate exact chunked nearest-neighbor anomaly scores;
- derive thresholds only from held-out normal images;
- evaluate complete MVTec AD category test partitions;
- export and reload versioned model artifacts;
- classify images from paths or binary streams;
- return decisions and threshold-normalized PNG heatmaps through FastAPI;
- serve a public client-neutral API through ASP.NET Core;
- display interactive heatmap overlays through a WPF desktop client;
- run the backend and inference service through Docker Compose.

The web client, pixel-level localization metrics, production deployment hardening, multi-category runtime selection, and public artifact distribution remain future work.

## Model Approach

The current PatchCore-inspired pipeline is:

```text
normal source images
-> recursive image discovery
-> deterministic fitting and validation split
-> direct square resize and ImageNet normalization
-> frozen pretrained ResNet18
-> layer2 and layer3 feature maps
-> 384-dimensional multi-scale patch embeddings
-> normal feature memory
-> exact chunked nearest-neighbor search
-> patch anomaly scores
-> top-fraction image-score aggregation
-> normal-validation-derived threshold
-> versioned artifact
-> normal/anomalous decision and anomaly heatmap
```

The fitting process does not fine-tune ResNet18. It constructs the feature memory from normal fitting images and derives the decision threshold from held-out normal validation images. One artifact represents one product category and must not be assumed to generalize to unrelated categories.

## Verified Development Environment

| Component | Verified value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| FastAPI | 0.139.2 |
| Uvicorn | 0.50.0 |
| Pillow | 12.2.0 |
| NumPy | 2.4.4 |
| ONNX Runtime | 1.28.0 |
| .NET SDK | 10.0.400 |
| Git | 2.55.0.windows.3 |
| Docker Engine | 29.6.1 |
| Docker Compose | 5.3.0 |

The Python version is recorded in `.python-version`, and direct dependencies are pinned in `requirements.txt`. Source compilation and `pip check` succeed. The Python suite currently contains 92 passing tests.

The FastAPI test client emits a third-party Starlette deprecation warning concerning HTTPX. It does not fail the suite and should be handled during a focused dependency update.

## Dataset and Artifact Policy

Datasets are stored outside Git. Dataset archives, extracted images, validation reports, generated split sidecars, feature memories, artifacts, and experiment outputs are excluded from the repository.

The locally acquired MVTec AD, MVTec LOCO AD, and MVTec AD 2 datasets pass the implemented structure, readability, inventory, mask-name, and mask-content checks. Private MVTec AD 2 ground truth is not stored locally.

Generated generalized splits use relative paths and are validated for counts, duplicates, overlap, and unsafe traversal. Defect images are not used for feature-memory construction or threshold selection.

## Generalized Directory-Based Fitting

The model can now be fitted without an MVTec-specific manifest. The generalized exporter accepts one directory containing normal images:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_image_directory_model.py `
    --image-directory C:\path\to\normal-images `
    --dataset custom-dataset `
    --category product-category `
    --output-directory .\outputs\model-artifacts\custom-category-320 `
    --validation-fraction 0.2 `
    --split-seed 42 `
    --input-size 320 `
    --top-fraction 0.01 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

The exporter:

1. discovers `.png`, `.jpg`, and `.jpeg` files recursively and case-insensitively;
2. rejects missing, empty, or invalid source directories;
3. creates non-overlapping deterministic fitting and validation partitions;
4. builds the feature memory from fitting images;
5. derives the threshold from validation images;
6. exports `metadata.json` and `feature_memory.pt`;
7. writes exact split membership to `training_split.json` with relative paths.

The established `export_mvtec_ad_model.py` workflow remains supported and delegates to the same reusable training implementation.

## Generalized Bottle Verification

The first generalized export used the normal MVTec AD Bottle training directory as a generic image collection. The new exporter did not use an MVTec manifest or interpret the MVTec directory structure.

| Property | Verified value |
| --- | --- |
| Dataset metadata | `mvtec-ad` |
| Category | `bottle` |
| Source normal images | 209 |
| Fitting images | 167 |
| Validation images | 42 |
| Validation fraction | 0.2 |
| Split seed | 42 |
| Input size | 320 x 320 |
| Patch grid | 40 x 40 |
| Embedding dimension | 384 |
| Feature-memory entries | 267,200 |
| Feature-memory size | 391.41 MiB |
| Aggregation | `top_fraction_mean` |
| Top fraction | 0.01 |
| Threshold | 3.2163138389587402 |

Direct inference smoke tests produced:

| Image | Score | Threshold | Decision |
| --- | ---: | ---: | --- |
| Bottle `test/good/000.png` | 2.613894 | 3.216314 | normal |
| Bottle `test/broken_large/000.png` | 4.907190 | 3.216314 | anomalous |

These two predictions verify artifact compatibility and basic behavior. They do not replace a complete category evaluation.

## Backward-Compatibility Verification

The refactored manifest-based exporter was run with the established Capsule split and 320 x 320 configuration. It reproduced:

- 175 fitting and 44 validation images;
- a 280,000 x 384 feature memory;
- a 410.16 MiB feature-memory file;
- threshold `2.501821517944336`;
- identical metadata;
- the established feature-memory SHA-256 hash:

```text
51DE3F2B4FEF804E9E95900597E738E86F7044A669D2739956CBA0CC6DE65478
```

This confirms byte-for-byte preservation of the established Capsule feature memory through the refactoring.

## Exploratory Reference Results

Bottle at 224 x 224 with complete feature memory:

| Aggregation | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum | 0.9398 | 1.0000 | 0.9206 | 0.9587 | 0 | 5 |
| Top 1% mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Top 5% mean | 0.9759 | 0.9692 | 1.0000 | 0.9844 | 2 | 0 |

Capsule at 320 x 320 with complete feature memory and top-one-percent aggregation:

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

These findings are exploratory because test images were inspected during development. They are not untouched final benchmarks.

## Feature-Memory Sampling Experiment

Deterministic sampling with seed `42` was evaluated for Capsule at 320 x 320:

| Memory | Entries | Validation + test scoring | Recall | F1 | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 280,000 | 225.06 s | 0.9541 | 0.9674 | 5 |
| 75% | 210,000 | 184.33 s | 0.8991 | 0.9423 | 11 |
| 50% | 140,000 | 120.36 s | 0.8624 | 0.9216 | 15 |
| 25% | 70,000 | 64.15 s | 0.6330 | 0.7753 | 40 |

Sampling accelerates search but degrades recall too strongly. Complete memory remains the quality baseline.

## Artifact Format

The manifest-based exporter writes:

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

The generalized exporter additionally writes:

```text
model-artifact/
  metadata.json
  feature_memory.pt
  training_split.json
```

Metadata records dataset, category, backbone, input and patch-grid sizes, embedding dimension, aggregation, threshold, sampling configuration, and feature-memory entry count. The split sidecar records seed, requested validation fraction, counts, and exact fitting and validation membership.

## Inference and Application Integration

The internal FastAPI service loads one configured artifact and feature extractor at startup. It provides:

```text
GET  /health/live
POST /api/v1/predictions
```

Predictions contain model identity, score, threshold, decision, and a threshold-normalized Base64-encoded RGB PNG heatmap.

The following full workflow has been verified locally:

```text
WPF desktop client
-> ASP.NET Core POST /api/v1/analyses
-> FastAPI POST /api/v1/predictions
-> loaded PyTorch artifact
-> score, threshold, decision, and heatmap
-> ASP.NET Core response
-> interactive WPF heatmap overlay
```

The desktop client displays normal and anomalous decisions, model details, timing, trace identifiers, and a heatmap overlay with visibility and opacity controls.

## Docker Stack Verification

The separate `industrial-visual-anomaly-detection-stack` repository builds version-pinned Linux images for the Python inference service and ASP.NET Core backend. Docker Compose supplies service networking, health checks, startup ordering, read-only artifact mounting, and host port publication.

A clean-clone verification confirmed inference liveness, backend liveness, backend readiness, real anomalous Capsule analysis, 320 x 320 PNG heatmap transport, teardown, and restart. The WPF client remains a native Windows process and connects to the published backend port.

## Automated Tests and Quality Checks

The Python repository has 92 passing tests covering deterministic model components, artifacts, inference, image discovery, deterministic splitting, split-manifest persistence, generalized training invariants, service configuration, runtime lifecycle, FastAPI behavior, upload validation, and heatmap encoding.

The following checks pass:

```text
unittest discovery
compileall
pip check
git diff --check
```

Dataset-dependent evaluation, large artifact exports, real service startup, and cross-repository integration remain manual checks because licensed datasets and generated feature memories are not stored in Git.

## Current Repository Shape

```text
scripts/
  create_mvtec_ad_split.py
  evaluate_mvtec_ad_category.py
  export_image_directory_model.py
  export_mvtec_ad_model.py
  predict_image.py
  validate_mvtec_ad.py
  validate_mvtec_ad_2.py
  validate_mvtec_loco_ad.py
src/
  industrial_visual_anomaly_detection/
    artifacts/
    datasets/
      image_discovery.py
      image_split.py
      image_split_manifest.py
    models/
    service/
    evaluation.py
    inference.py
    preprocessing.py
    training.py
    visualization.py
tests/
```

## Confirmed Decisions

- normal-only unsupervised fitting;
- one artifact per product category;
- frozen pretrained ResNet18;
- `layer2` and `layer3` feature fusion;
- 384-dimensional embeddings;
- direct square resizing and ImageNet normalization;
- exact chunked nearest-neighbor search;
- complete feature memory as the quality baseline;
- top-one-percent mean aggregation;
- maximum normal-validation score as threshold;
- deterministic external-directory splits with recorded manifests;
- Python as the authoritative inference runtime;
- FastAPI as the internal service boundary;
- ASP.NET Core as the public client-neutral API;
- ONNX as an optional future path rather than an integration prerequisite.

## Open Decisions

- selection and lifecycle of multiple category artifacts;
- preprocessing for non-square categories;
- minimum recommended normal-image count for external datasets;
- validation strategy for small datasets;
- coverage-preserving feature-memory reduction;
- approximate nearest-neighbor search;
- quantitative pixel-level metrics and map smoothing;
- stronger artifact provenance, preprocessing metadata, and checksums;
- public artifact release contents;
- future threshold calibration;
- timeout, retry, cancellation, and concurrency policies.

## Deferred Work

- web client;
- database persistence;
- authentication and authorization;
- production deployment hardening;
- GPU optimization;
- backbone fine-tuning;
- multi-category runtime orchestration;
- defect-type classification;
- private MVTec AD 2 benchmark submission;
- production monitoring and drift handling;
- regulatory or industrial validation.

## Immediate Next Steps

1. Evaluate the generalized workflow on a non-MVTec normal-image collection.
2. Define the external dataset contract and minimum useful image counts.
3. Add artifact evaluation against optional normal and anomalous test directories.
4. Decide how multiple category artifacts are selected by the inference service and backend.
5. Strengthen artifact provenance, preprocessing metadata, and checksums.
6. Investigate coverage-preserving memory reduction and faster nearest-neighbor search.
7. Update release documentation after the generalized training milestone is finalized.

## Last Verified Status

As of 2026-08-19:

- 92 Python tests pass;
- compilation, dependency, and whitespace checks pass;
- normal PNG and JPEG images can be discovered recursively;
- deterministic fitting and validation splits are independent of MVTec;
- exact split membership is recorded in `training_split.json`;
- a 320 x 320 Bottle artifact was fitted directly from a normal-image directory;
- known normal and defective Bottle images produced the expected decisions;
- the established Capsule artifact remains byte-for-byte reproducible;
- the FastAPI service returns anomaly decisions and heatmaps;
- the backend forwards the verified analysis contract;
- the desktop client displays interactive heatmap overlays;
- the Docker Compose stack runs the versioned backend and inference service from a clean clone;
- generated datasets and artifacts remain excluded from Git.

The next active milestone is validating the generalized workflow on genuinely external data, followed by explicit multi-category artifact selection.
