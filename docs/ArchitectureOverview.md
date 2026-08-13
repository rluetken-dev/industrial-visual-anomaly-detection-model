# Industrial Visual Anomaly Detection – Architecture Overview

## Purpose

This document describes the implemented and intended architecture of the Industrial Visual Anomaly Detection system.

The architecture separates:

- external dataset storage and validation;
- Python model development, evaluation, export, and reference inference;
- versioned model artifacts;
- a future client-neutral .NET inference backend;
- future web and desktop clients.

The Python reference pipeline is now implemented end to end. It can fit a normal feature memory, select a validation threshold, evaluate MVTec AD categories, export and load model artifacts, and classify individual images. The .NET backend and client applications remain target architecture rather than implemented components.

## Architecture Status

This document uses three status categories:

- **Implemented and verified** – demonstrated through executable code, automated tests, or recorded experiments;
- **Selected direction** – established as the current target but not yet complete across runtimes;
- **Open** – requires further design, implementation, or evaluation.

### Implemented and Verified

- Python 3.12 CPU-based model development environment;
- validation of MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- optional schema-versioned JSON validation reports;
- deterministic category-specific fitting and validation manifests;
- configurable square preprocessing with ImageNet normalization;
- frozen pretrained ResNet18 feature extraction;
- fusion of `layer2` and `layer3` feature maps;
- local patch-embedding creation;
- complete normal feature-memory construction;
- exact chunked nearest-neighbor scoring;
- maximum and top-fraction image-score aggregation;
- threshold selection from normal validation images;
- grouped image-level evaluation;
- anomaly-map and heatmap generation;
- deterministic random feature-memory sampling experiments;
- model-artifact export and loading;
- single-image inference and a prediction CLI;
- provisional ONNX feature-extractor export and PyTorch/ONNX parity;
- 54 passing automated tests.

### Selected Direction

- the current reference model is MVTec AD Capsule at 320 × 320;
- the reference configuration uses complete feature memory and top-one-percent mean aggregation;
- Python remains the reference implementation for fitting, evaluation, and artifact production;
- ONNX is the intended neural feature-extractor format for .NET interoperability;
- a future ASP.NET Core backend should expose one client-neutral inference contract;
- future React and WPF clients should consume the same backend;
- model, backend, web, and desktop concerns should remain independently versioned where practical.

### Open

- framework-neutral feature-memory serialization;
- complete preprocessing metadata in the artifact contract;
- packaging ONNX, feature memory, metadata, checksums, and evaluation results;
- approximate nearest-neighbor or coverage-preserving coreset optimization;
- pixel-level metric selection and anomaly-map post-processing;
- Python/.NET numerical parity for the complete pipeline;
- backend persistence, authentication, deployment, and artifact distribution.

## Architectural Goals

The architecture should:

- keep model research independent from presentation technology;
- keep datasets and generated artifacts outside source control;
- make splits, fitting, thresholds, and experiments reproducible;
- preserve preprocessing and scoring semantics across runtimes;
- support image-level detection and patch-level localization;
- expose stable, client-neutral results;
- keep CPU inference viable;
- make artifacts versioned, traceable, and compatible with their runtime;
- prevent test data from influencing fitting or threshold selection;
- support future model replacement without rewriting clients;
- distinguish an experimental benchmark system from a validated industrial inspection system.

## System Context

```text
External datasets
        ↓
Python model-development repository
        ↓
Evaluated model artifact
        ↓
Future .NET inference backend
        ↓
Client-neutral API
        ↓
React web client      WPF desktop client
```

The current repository implements the Python portion and a local artifact format. It does not yet provide the .NET runtime or client applications.

## Repository Boundaries

The preferred long-term separation is:

```text
industrial-visual-anomaly-detection-model
industrial-visual-anomaly-detection-backend
industrial-visual-anomaly-detection-web
industrial-visual-anomaly-detection-desktop
```

Only the model repository currently exists.

### Model Repository

The model repository owns:

- dataset documentation and validation;
- deterministic split creation;
- preprocessing experiments and implementation;
- feature extraction and patch embeddings;
- feature-memory creation and experiments;
- anomaly scoring, threshold selection, and evaluation;
- heatmap generation;
- ONNX feasibility and parity checks;
- model-artifact export, loading, and reference inference;
- experiment and artifact metadata design.

It must not own ASP.NET Core hosting, WPF or React presentation code, production authentication, or operational database persistence.

### Future Backend Repository

The backend should own:

- artifact discovery, validation, and persistent loading;
- production preprocessing equivalent to Python;
- ONNX Runtime integration;
- patch-embedding construction;
- feature-memory search and score aggregation;
- threshold application and anomaly-map production;
- stable request and response contracts;
- optional inspection-history persistence;
- health, diagnostics, security controls, and observability.

### Future Web and Desktop Repositories

Clients should own image selection, previews, request submission, result visualization, history views, platform-specific interaction, and presentation tests. They must not duplicate preprocessing, scoring, or threshold rules.

The first WPF client should consume the backend API. Offline desktop inference remains optional future work.

## Current Model Repository Structure

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
.gitignore
.python-version
environment_check.py
pyproject.toml
README.md
requirements.txt
```

Virtual environments, datasets, reports, caches, ONNX exports, model artifacts, and experiment outputs are excluded from version control.

## Python Model Architecture

```text
validated normal images
        ↓
deterministic fitting and validation manifest
        ↓
category configuration and preprocessing
        ↓
frozen pretrained ResNet18
        ↓
layer2 and layer3 feature maps
        ↓
384-dimensional patch embeddings
        ↓
complete or sampled normal feature memory
        ↓
exact chunked nearest-neighbor distances
        ↓
patch-score grid
        ↓
image-score aggregation
        ↓
normal-validation threshold
        ↓
decision, evaluation, heatmap, and artifact export
```

### Dataset Boundary

Datasets are stored outside the repository under:

```text
C:/dev/data/industrial-visual-anomaly-detection/
```

Runtime commands receive the local root explicitly. Public configuration and manifests do not contain machine-specific dataset roots.

Validators check relevant combinations of expected structure, inventories, image readability, dimensions, modes, mask naming, mask dimensions, and mask content. Successful validation can produce ignored local JSON reports.

### Split Manifests

Versioned manifests define deterministic membership with relative paths. The current manifests are:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
configs/splits/mvtec-ad-capsule-seed-42.json
```

Manifest loading validates counts, duplicates, overlap, absolute paths, and parent traversal. Paths are resolved against a dataset root supplied at runtime.

### Preprocessing

The implemented preprocessing performs:

- image decoding and RGB conversion;
- direct resizing to a configured square input;
- bilinear interpolation with antialiasing;
- tensor conversion;
- ImageNet normalization.

Bottle uses 224 × 224. The selected Capsule reference uses 320 × 320. Direct resizing was selected for Bottle because center cropping removed object-boundary information.

The artifact currently stores the input size. Mean, standard deviation, interpolation, antialiasing, and RGB conversion remain defined by the versioned Python implementation. A cross-runtime artifact must store these semantics explicitly.

### Feature Extractor and Patch Embeddings

For 224 × 224 input:

```text
layer2:          (1, 128, 28, 28)
layer3:          (1, 256, 14, 14)
patch embeddings:      (784, 384)
```

For 320 × 320 input:

```text
layer2:          (1, 128, 40, 40)
layer3:          (1, 256, 20, 20)
patch embeddings:     (1600, 384)
```

`layer3` is resized to the spatial resolution of `layer2`, concatenated along the channel dimension, and rearranged into one 384-dimensional embedding per patch position. The backbone is frozen and kept in evaluation mode.

### Feature Memory

The feature memory contains normal fitting embeddings and is part of the fitted model state. It must never be combined with incompatible preprocessing, backbone weights, feature layers, dimensions, or input resolution.

The selected Capsule artifact contains:

```text
shape: (280000, 384)
dtype: float32
size:  approximately 410.16 MiB
```

Deterministic random sampling at 75%, 50%, and 25% reduced runtime but degraded recall. Complete memory remains the reference configuration. Smarter coreset selection is open.

### Nearest-Neighbor Scoring

For every query patch, the implementation computes its Euclidean distance to the nearest normal feature-memory entry. Memory is processed in configurable chunks to bound temporary distance tensors while preserving exact results.

Distances are reshaped into a patch grid. Supported image aggregation includes the maximum patch score and the mean of the highest configurable fraction. Top-one-percent mean is selected for the Capsule reference artifact.

### Threshold and Evaluation

The threshold is selected exclusively from normal validation images as their maximum image-level score. Only scores strictly above the threshold are anomalous.

Official test labels and masks are not used for fitting or threshold calculation. Test-folder names support grouped analysis. The current metrics include confusion-matrix counts, accuracy, precision, recall, F1, per-group detection rates, and runtime measurements.

Bottle and Capsule results are exploratory because their test partitions were inspected during development. Pixel-level evaluation remains open.

### Visualization

Patch grids can be resized to image resolution, normalized, colorized, and blended with source images. Fixed threshold-based normalization provides more comparable heatmaps than independent per-image normalization.

Visualization is separate from classification logic. A heatmap supports interpretation but is not evidence of localization accuracy without mask-based metrics.

## Model Artifact Architecture

### Implemented Local Artifact

The current artifact directory contains:

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

The typed metadata contains:

- schema version;
- dataset and category;
- backbone name;
- input and patch-grid sizes;
- embedding dimension;
- aggregation method and top fraction;
- validation-derived threshold;
- memory fraction and sampling seed;
- feature-memory entry count.

The writer validates shape, finite values, entry count, and embedding dimension. The loader verifies the required files, reconstructs the typed metadata, loads the tensor on CPU with `weights_only=True`, and validates the tensor type, dimensionality, finite values, entry count, and embedding dimension.

The local format is sufficient for Python reference inference but is not yet framework-neutral.

### Reference Capsule Artifact

```text
dataset:                mvtec-ad
category:               capsule
backbone:               resnet18
input size:             320 × 320
patch grid:             40 × 40
embedding dimension:    384
feature-memory entries: 280000
aggregation:            top_fraction_mean
top fraction:           0.01
memory fraction:        1.0
threshold:              2.501821517944336
```

The generated artifact is stored under `outputs/model-artifacts/` and ignored by Git.

### Intended Cross-Runtime Package

A future release package should evolve toward:

```text
model-package/
  feature-extractor.onnx
  feature-memory.<framework-neutral-format>
  model-metadata.json
  evaluation-summary.json
  checksums.txt
  notices/
```

Metadata must additionally identify the pretrained weight version, feature layers, complete preprocessing semantics, split manifest, software versions, artifact version, license attribution, and evaluation context.

The consuming runtime must reject unsupported schema versions, missing files, incompatible dimensions, and invalid checksums.

## Reference Inference Architecture

The implemented Python inference flow is:

```text
artifact directory + image
        ↓
load metadata and feature memory
        ↓
create frozen ResNet18 extractor
        ↓
decode, convert, resize, and normalize image
        ↓
extract patch embeddings
        ↓
exact nearest-neighbor scoring
        ↓
top-fraction aggregation
        ↓
compare score with stored threshold
        ↓
AnomalyPrediction
```

`AnomalyPrediction` contains the resolved image path, score, threshold, Boolean decision, and patch-score map. `scripts/predict_image.py` exposes this flow through a CLI.

Verified Capsule predictions took approximately 1.44–1.46 seconds each on the current CPU, excluding the separately measured artifact load and extractor creation time. A persistent service should initialize the artifact and extractor once and reuse them.

## Future .NET Runtime Architecture

```text
ASP.NET Core API
        ↓
application orchestration
        ↓
inspection domain model
        ↓
model runtime abstraction
        ↓
ONNX Runtime + feature memory + scoring
```

### Domain

The domain should represent inspection requests and results, anomaly decisions, scores, patch maps or result references, model identity, and validation failures. It must not depend on ASP.NET Core, WPF, React, ONNX Runtime, or a database provider.

### Application

The application layer should coordinate request validation, decoding, model selection, inference, optional persistence, and response creation.

### Model Runtime

The runtime should own artifact validation, persistent loading, ONNX session creation, preprocessing, feature-output validation, embedding construction, nearest-neighbor search, aggregation, heatmaps, and threshold application. It should be independently testable without API hosting.

### Infrastructure

Infrastructure may provide file-system artifact storage, inspection-history persistence, image storage, identifiers, clocks, logging, and configuration.

### API

The API should provide client-neutral operations for inspecting an image, retrieving model information, accessing optional history, and reporting health and readiness.

## Intended Request Flow

```text
client submits image
        ↓
API validates payload and selects compatible artifact
        ↓
runtime decodes and preprocesses image
        ↓
ONNX extractor produces feature maps
        ↓
runtime creates patch embeddings
        ↓
nearest-neighbor search produces patch scores
        ↓
aggregation and threshold produce decision
        ↓
application creates client-neutral response
        ↓
optional persistence records inspection metadata
        ↓
client renders decision, score, and overlay
```

## Client-Neutral Result Contract

Appropriate fields include:

- inspection identifier;
- model and artifact version;
- category;
- anomaly score and threshold;
- normal/anomalous decision;
- anomaly-map or overlay reference;
- processing duration;
- warnings and validation messages.

The backend must not return UI-specific instructions such as colors, tab selection, dialogs, WPF commands, or React component names.

## Configuration and Persistence

Configuration must separate public defaults, local overrides, deployment values, and secrets. Reproducibility-critical model behavior belongs in artifact metadata rather than only environment configuration.

Optional inspection persistence may record identifiers, timestamps, model versions, image references or checksums, scores, thresholds, decisions, durations, and failures. Raw image retention must be optional and governed by explicit privacy and retention rules.

## Security and Safety

The future backend must treat images and artifacts as untrusted inputs. Required safeguards include media validation, bounded upload and decoded dimensions, controlled temporary storage, cancellation, timeouts, schema validation, checksum validation, and safe errors.

The project must not be represented as a validated production quality-control system without domain validation, controlled deployment, regulatory assessment where applicable, and operational monitoring.

## Testing Strategy

### Implemented Python Tests

The current 54 tests cover manifests, dataset discovery, preprocessing, embeddings, feature memory, sampling, nearest-neighbor distances, scoring, aggregation, metrics, visualization, and artifact persistence.

### Additional Python Tests

Next coverage should include single-image inference behavior, artifact schema compatibility, malformed metadata, CLI result serialization, and experiment-report generation.

### Future Cross-Runtime Tests

Fixed fixtures should compare Python and .NET results for preprocessing tensors, ONNX features, patch embeddings, nearest-neighbor distances, aggregated scores, maps, and threshold decisions.

### Future API and Client Tests

API tests should cover valid requests, malformed images, unsupported media, payload limits, unavailable artifacts, cancellation, and stable errors. Client tests should cover selection, submission, loading, success, error, history, and overlay rendering.

## Observability and Deployment Direction

The future backend should record artifact identity, preprocessing and inference duration, failures, outcomes, and resource use without exposing sensitive image content.

Initial deployment should remain CPU-compatible. Potential modes include a self-contained .NET backend, central API hosting, WPF and React clients, and later optional offline desktop inference after parity is proven.

## Known Architectural Risks

- complete feature memories are large;
- exact CPU nearest-neighbor search may limit throughput;
- random sampling reduces quality on Capsule;
- preprocessing mismatch can invalidate cross-runtime results;
- small defects may require higher input resolution;
- non-square categories need a deliberate resize policy;
- threshold or hyperparameter selection can leak test information;
- ONNX represents the neural extractor, not automatically the complete pipeline;
- artifact evolution requires schema and compatibility management;
- dataset licenses constrain redistribution and commercial use;
- benchmark results may not transfer to real industrial imagery;
- a prototype may be mistaken for a validated inspection system.

## Current Non-Goals

- production deployment;
- real-time camera or PLC integration;
- authentication and a production database;
- implemented web or desktop UI;
- supervised defect-type classification;
- automated retraining or multi-model orchestration;
- regulatory validation;
- real pharmaceutical packaging validation.

## Completed Architectural Milestones

1. Reproducible Python environment and dependency setup.
2. CPU-based pretrained ResNet18 and intermediate feature extraction.
3. Patch embeddings and ONNX feasibility with parity checks.
4. Acquisition and validation of three MVTec datasets.
5. Machine-readable dataset reports.
6. Deterministic Bottle and Capsule manifests.
7. Reusable dataset, preprocessing, model, scoring, evaluation, and visualization modules.
8. Bottle baseline and Capsule generalization evaluation.
9. Configurable resolution and aggregation experiments.
10. Feature-memory sampling implementation and tradeoff evaluation.
11. Typed artifact metadata, writer, loader, and round-trip tests.
12. Export of the complete 320 × 320 Capsule reference artifact.
13. Verified normal and anomalous single-image CLI inference.

## Immediate Architectural Steps

1. Consolidate the remaining strategy, specification, and README documents.
2. Add inference and artifact-compatibility tests.
3. Define the complete framework-neutral artifact schema.
4. Export and verify the selected 320 × 320 ONNX extractor.
5. Select a framework-neutral feature-memory format.
6. Add checksums and machine-readable evaluation summaries.
7. Verify full Python artifact reproducibility from a clean environment.
8. Build a minimal .NET console parity spike before designing the web API.
9. Finalize the backend boundary only after parity is demonstrated.
10. Add clients after the backend contract is stable.

## Related Documentation

- `DevelopmentStatus.md` records verified results and active work.
- `ProjectSpecification.md` defines product scope and requirements.
- `ModelDevelopmentStrategy.md` defines fitting, validation, and experiment rules.
- `DatasetDocumentation.md` records sources, licenses, structures, and validation.
- A future `ModelCard.md` should document a released evaluated artifact.

## Last Updated

This architecture reflects the verified project state as of 2026-08-13.
