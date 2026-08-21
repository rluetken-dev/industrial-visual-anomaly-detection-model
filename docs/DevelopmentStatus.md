# Industrial Visual Anomaly Detection – Development Status

## Document Purpose

This living document records verified implementation progress, experiment results, current decisions, open questions, and immediate next steps for the Industrial Visual Anomaly Detection model project.

It distinguishes between automated verification, manually verified integration behavior, exploratory benchmark findings, and planned work.

## Current Phase

The project has completed its initial model-development, generalized artifact-export, evaluation, inference-service, backend-integration, desktop-client, heatmap-visualization, Docker Compose, and registry-based multi-model milestones.

The inference service can now start from a validated model registry, load multiple enabled artifacts, expose their catalog, use a configured default model, and select an explicit model per prediction request. The earlier single-artifact environment variable remains available as a compatibility mode.

The multi-model implementation is committed on `main` in commit:

```text
e94eb01 feat(service): support configurable model registry
```

The registry-capable source has been built into the local Docker stack and verified with a four-model catalog containing Capsule, Bottle, Candle, and Cashew. Containerized prediction requests explicitly selected Capsule and Cashew without recreating the inference container. Native integration additionally exercised Bottle and Candle.

The next release milestone is to finish documentation and publish an immutable registry-capable model-service release. Independent threshold validation, stronger provenance, and runtime memory optimization remain subsequent model-quality work.

## Implemented System Capabilities

- validate locally acquired MVTec datasets;
- discover normal PNG and JPEG images recursively;
- create deterministic fitting and validation partitions;
- record partitions with relative paths in `training_split.json`;
- extract frozen multi-scale ResNet18 features;
- build category-specific normal feature memories;
- calculate exact chunked nearest-neighbor anomaly scores;
- derive configurable quantile thresholds from held-out normal images;
- evaluate exported artifacts through labeled CSV manifests;
- export and reload schema-versioned artifacts;
- load schema-version-1 artifacts through compatibility defaults;
- classify images from paths or binary streams;
- generate threshold-normalized PNG heatmaps;
- start FastAPI from one legacy artifact;
- parse and validate a multi-model registry;
- load every enabled registry artifact during startup;
- expose the enabled model catalog and configured default;
- select a model explicitly for each prediction request;
- fall back to the configured default when no model is supplied;
- serve model-specific predictions to ASP.NET Core;
- support dynamic model selection in the WPF client through the backend;
- run the registry-based inference service and backend through Docker Compose.

The web client, pixel-level metrics, production hardening, dynamic registry reload, lazy loading, and public artifact distribution remain future work.

## Model Approach

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
-> normal-validation quantile threshold
-> versioned artifact
-> normal/anomalous decision and anomaly heatmap
```

Fitting does not fine-tune ResNet18. One artifact represents one product category and must not be assumed to generalize to unrelated categories. Multi-model serving coordinates several independent category-specific artifacts; it does not merge their feature memories.

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

The Python version is recorded in `.python-version`, and direct dependencies are pinned in `requirements.txt`. The suite currently contains 144 automated test methods.

## Dataset and Artifact Policy

Datasets, archives, extracted images, reports, feature memories, registries, artifacts, and experiment outputs remain outside Git.

The locally acquired MVTec AD, MVTec LOCO AD, and MVTec AD 2 datasets pass the implemented structural checks. Private MVTec AD 2 ground truth is not stored locally.

The official VisA archive was verified locally with SHA-256:

```text
2EB8690C803AB37DE0324772964100169EC8BA1FA3F7E94291C9CA673F40F362
```

Generated splits use relative paths and reject duplicates, overlap, and unsafe traversal. Defect images are excluded from feature-memory construction and initial threshold selection.

## Generalized Directory-Based Fitting

The generalized exporter accepts a directory containing normal images:

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
    --threshold-quantile 0.95 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

It discovers supported images, creates non-overlapping partitions, builds memory from fitting images, derives a validation threshold, exports `metadata.json` and `feature_memory.pt`, and records membership in `training_split.json`.

The established MVTec manifest workflow delegates to the same reusable training implementation.

## Dataset-Independent Evaluation

The evaluator accepts an artifact, dataset root, and CSV manifest with `image`, `group`, and `is_anomalous`:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_model_artifact.py `
    --artifact .\outputs\model-artifacts\visa-candle-generalized-q95-320 `
    --dataset-root C:\path\to\visa\extracted `
    --manifest C:\path\to\visa\evaluation_manifest.csv
```

It reports score distributions, confusion-matrix counts, accuracy, precision, recall, specificity, F1, group rates, false positives, and false negatives.

## VisA Candle Threshold Calibration

The official one-class split contains 900 normal training images, 100 normal test images, and 100 anomalous test images. The generalized exporter divided the 900 training images deterministically into 720 fitting and 180 validation images.

| Property | Verified value |
| --- | --- |
| Input size | 320 × 320 |
| Patch grid | 40 × 40 |
| Embedding dimension | 384 |
| Complete fitting memory | 1,152,000 × 384 |
| Exported feature memory | 288,000 × 384 |
| Memory fraction | 0.25 |
| Sampling seed | 42 |
| Top fraction | 0.01 |
| Split seed | 42 |

The q100, q99, and q95 artifacts have identical feature-memory hashes. Only thresholds differ.

| Variant | Quantile | Threshold | Accuracy | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q100 | 1.00 | 3.373678 | 0.6550 | 1.0000 | 0.3100 | 1.0000 | 0.4733 |
| q99 | 0.99 | 3.001366 | 0.7800 | 1.0000 | 0.5600 | 1.0000 | 0.7179 |
| q95 | 0.95 | 2.763051 | 0.8200 | 0.9324 | 0.6900 | 0.9500 | 0.7931 |

q95 is a provisional review-oriented candidate. Because official test images were inspected during comparison, it is exploratory calibration rather than an independent final estimate. Further tuning against the same split is excluded.

## Generalized Bottle Verification

The first generalized export used the MVTec AD Bottle normal training directory without interpreting MVTec structure.

| Property | Verified value |
| --- | --- |
| Source normal images | 209 |
| Fitting images | 167 |
| Validation images | 42 |
| Input size | 320 × 320 |
| Feature-memory entries | 267,200 |
| Aggregation | `top_fraction_mean` |
| Threshold | 3.2163138389587402 |

Smoke tests classified a known normal image as normal and a known broken image as anomalous. This verifies compatibility and basic behavior, not a complete category benchmark.

## Artifact Compatibility Verification

The refactored Capsule exporter reproduced 175 fitting and 44 validation images, a 280,000 × 384 feature memory, threshold `2.501821517944336`, identical metadata, and the established feature-memory hash:

```text
51DE3F2B4FEF804E9E95900597E738E86F7044A669D2739956CBA0CC6DE65478
```

This confirms byte-for-byte preservation of the Capsule memory through the training refactor.

## Model Registry Implementation

The service registry uses this conceptual layout:

```text
model-artifacts/
  models.json
  mvtec-ad-capsule-320/
  mvtec-ad-bottle-generalized-320/
  visa-candle-generalized-q95-320/
  visa-cashew-generalized-q95-320/
```

`models.json` records schema version, default model, ordered model entries, stable IDs, display names, relative artifact directories, and enabled states.

Implemented validation covers:

- missing or non-object documents;
- unsupported schema versions;
- unexpected fields;
- empty model arrays;
- duplicate IDs and artifact directories;
- invalid identifiers;
- absolute paths and parent traversal;
- missing enabled artifact directories;
- registries without enabled models;
- disabled or missing default models;
- UTF-8 BOM compatibility.

Disabled entries are not loaded or exposed and may reference absent artifacts.

## Runtime Registry Implementation

`InferenceRuntimeRegistry` loads one runtime per enabled entry and verifies that loaded runtime IDs match the configuration. It exposes models in configured order, derives catalog metadata from the artifacts, resolves explicit IDs, and falls back to the configured default.

Each runtime owns one loaded artifact, one frozen extractor, and one request lock. All enabled runtimes are loaded during application startup. Dynamic reload, lazy loading, and unloading are not implemented.

## FastAPI Service Integration

Two mutually exclusive configuration modes are supported:

```text
IVAD_MODEL_REGISTRY
IVAD_MODEL_ARTIFACT
```

Registry mode enables multi-model operation. The earlier `IVAD_MODEL_ARTIFACT` mode remains compatible with one artifact. Exactly one source is required.

Implemented endpoints:

```text
GET  /health/live
GET  /api/v1/models
POST /api/v1/predictions
```

The catalog endpoint returns the default and enabled models. Legacy mode appears as a one-model catalog.

Predictions accept multipart `image` and optional `modelId`. Registry mode selects the requested model or default. Unknown model IDs return not found. Responses identify the runtime actually used and include category, score, threshold, decision, and heatmap.

## Multi-Repository Integration Verification

The verified catalog contained:

```text
mvtec-ad-capsule-320
mvtec-ad-bottle-generalized-320
visa-candle-generalized-q95-320
visa-cashew-generalized-q95-320
```

Verified paths include:

- direct inference catalog retrieval;
- backend catalog mapping through `GET /api/v1/models`;
- desktop catalog loading and default selection;
- explicit desktop selection of Capsule, Bottle, Candle, and Cashew;
- model-specific analysis and heatmap display;
- Docker Compose startup with the registry and artifacts mounted read-only;
- explicit containerized Capsule and Cashew prediction requests;
- response model IDs matching the requested IDs;
- health, readiness, score, threshold, decision, trace, and 320 × 320 PNG heatmap verification.

This confirms that model switching occurs per request without editing `.env` or recreating containers.

## Automated Tests and Quality Checks

The Python repository contains 144 automated test methods covering:

- deterministic model components and training invariants;
- artifact persistence and compatibility;
- image discovery, splitting, and path safety;
- quantile thresholds and evaluation metrics;
- file and stream inference;
- registry configuration and invalid inputs;
- multi-runtime loading, catalog metadata, default selection, and errors;
- legacy and registry startup settings;
- FastAPI health, catalog, prediction, validation, and heatmap behavior.

Dataset-dependent evaluation, large exports, real artifact startup, and cross-repository integration remain manual because datasets and feature memories are not stored in Git.

## Confirmed Decisions

- normal-only unsupervised fitting;
- one artifact per product category;
- one registry per deployed model set;
- stable model IDs for request selection;
- one configured enabled default model;
- frozen pretrained ResNet18;
- `layer2` and `layer3` fusion;
- direct square resizing and ImageNet normalization;
- exact chunked nearest-neighbor search;
- complete feature memory as the quality baseline where practical;
- top-one-percent mean aggregation;
- configurable normal-validation quantile threshold;
- schema-version-2 metadata with schema-version-1 compatibility;
- Python as authoritative inference runtime;
- FastAPI as internal service boundary;
- ASP.NET Core as public client-neutral API;
- legacy single-artifact startup retained for compatibility;
- registry-based startup as the multi-model deployment path;
- eager loading of enabled models during startup;
- ONNX as an optional future path.

## Open Decisions

- preprocessing for non-square categories;
- minimum recommended normal-image counts;
- validation strategy for small datasets;
- coverage-preserving memory reduction;
- approximate nearest-neighbor search;
- lazy runtime loading and registry reload policy;
- pixel-level metrics and map smoothing;
- stronger provenance, metadata, and checksums;
- public artifact release contents;
- strict calibration and independent final-test protocol;
- production timeout, retry, cancellation, and concurrency policies.

## Deferred Work

- dynamic registry reload;
- lazy model loading and unloading;
- web client;
- database persistence;
- authentication and authorization;
- production deployment hardening;
- GPU optimization;
- backbone fine-tuning;
- automatic visual category recognition;
- defect-type classification;
- private MVTec AD 2 benchmark submission;
- production monitoring and drift handling;
- regulatory or industrial validation.

## Immediate Next Steps

1. Complete the registry-capable documentation update.
2. Run final Python quality checks and verify a clean intended diff.
3. Commit and push the documentation changes.
4. Verify CI for the documented registry implementation.
5. Publish an immutable registry-capable model-service release.
6. Update the stack to consume that release tag.
7. Continue independent threshold validation without retuning on inspected evidence.
8. Investigate provenance, integrity, and runtime memory improvements.

## Last Verified Status

As of 2026-08-21:

- registry support is committed on `main` at `e94eb01`;
- 144 automated Python test methods cover the current implementation;
- multiple enabled artifacts load from `models.json`;
- default and explicit runtime selection are implemented;
- `/api/v1/models` exposes the configured catalog;
- `/api/v1/predictions` accepts optional `modelId`;
- legacy single-artifact configuration remains supported;
- the four-model catalog has been verified across inference, backend, and desktop;
- Capsule, Bottle, Candle, and Cashew were exercised through the native application workflow;
- Capsule and Cashew were selected explicitly through the containerized stack;
- requested and returned model IDs matched;
- analysis results and PNG heatmaps were returned successfully;
- generated datasets, registries, and artifacts remain excluded from Git.

The next active milestone is publishing a registry-capable model-service release for immutable downstream stack builds.
