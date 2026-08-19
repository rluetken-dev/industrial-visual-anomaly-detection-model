# Industrial Visual Anomaly Detection – Architecture Overview

## Purpose

This document describes the implemented architecture, repository boundaries, data flow, model-fitting workflow, artifact contract, inference-service integration, and current evolution path of the Industrial Visual Anomaly Detection system.

The system consists of independently versioned repositories for:

- Python model development and inference;
- the ASP.NET Core public backend;
- the native WPF desktop client;
- Docker Compose orchestration;
- a possible future web client.

## Architecture Status

### Implemented and Verified

- validation of MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- recursive discovery of normal PNG and JPEG images from external directories;
- deterministic fitting and validation splits;
- portable split records with relative image paths;
- frozen ResNet18 multi-scale feature extraction;
- 384-dimensional patch embeddings;
- complete and optionally sampled feature memories;
- exact chunked nearest-neighbor scoring;
- validation-derived anomaly thresholds;
- category evaluation and exploratory metrics;
- anomaly-map and heatmap generation;
- reusable model-artifact export and loading;
- file-path and stream inference;
- internal FastAPI inference service;
- ASP.NET Core integration with health and error boundaries;
- WPF image selection, preview, analysis results, and interactive heatmap overlay;
- Docker Compose orchestration of the inference service and backend;
- 92 passing automated tests in the Python repository.

### Selected Direction

- Python remains authoritative for fitting, evaluation, artifact production, and model inference;
- ASP.NET Core remains the public client-neutral API boundary;
- clients call the backend rather than the internal Python service;
- one fitted artifact represents one product category;
- normal-image directories are the general fitting input;
- MVTec manifests remain a supported reproducible specialized workflow;
- HTTP remains the cross-runtime boundary;
- the WPF client remains native Windows software outside Docker;
- ONNX remains an optional portability path rather than a current integration requirement.

### Open

- multi-artifact loading and explicit category selection;
- evaluation on a genuinely non-MVTec image collection;
- external dataset conventions and minimum useful image counts;
- aspect-ratio handling for non-square products;
- stronger artifact provenance and preprocessing metadata;
- checksums and artifact integrity policies;
- approximate nearest-neighbor search or coverage-preserving memory reduction;
- quantitative pixel-level localization metrics;
- service authentication and production deployment hardening;
- a future web client.

## Architectural Goals

- keep datasets and generated artifacts outside source control;
- accept external normal-image collections without dataset-specific code;
- make discovery, splitting, fitting, thresholds, and artifacts reproducible;
- keep model logic independent from CLI, HTTP, and UI concerns;
- load large artifacts once per service process;
- keep the public API independent of Python and UI implementation details;
- make local startup reproducible through version-pinned container builds;
- preserve CPU-only execution;
- prevent test anomalies from influencing fitting or threshold selection;
- evolve artifact and HTTP contracts explicitly.

## System Context

```text
External normal-image collection
        |
        v
Python fitting and artifact export
        |
        v
Category-specific model artifact
        |
        v
Internal FastAPI inference service
        |
        v
ASP.NET Core public backend
        |
        +--> WPF desktop client
        |
        `--> future web client

Docker Compose stack
  -> builds and runs FastAPI + ASP.NET Core
  -> mounts the selected artifact read-only
  -> publishes backend and optional inference ports
```

## Repository Boundaries

```text
industrial-visual-anomaly-detection-model
industrial-visual-anomaly-detection-backend
industrial-visual-anomaly-detection-desktop
industrial-visual-anomaly-detection-stack
future web repository
```

### Model Repository

Owns:

- dataset validation and image discovery;
- deterministic split generation and recording;
- preprocessing and feature extraction;
- feature-memory construction and sampling;
- anomaly scoring, threshold selection, and evaluation;
- artifact export, validation, loading, and inference;
- heatmap generation and encoding;
- the internal FastAPI service;
- model experiments and documentation.

### Backend Repository

Owns:

- the public HTTP API;
- upload validation and size limits;
- Problem Details and trace identifiers;
- liveness and readiness semantics;
- HTTP communication with the Python service;
- mapping internal inference responses to client-neutral contracts;
- public configuration, logging, timeouts, and failure behavior.

### Desktop Repository

Owns:

- native Windows presentation;
- backend health and readiness display;
- image selection and preview;
- analysis submission and cancellation behavior;
- result presentation;
- interactive heatmap visibility and opacity;
- WPF-specific state, commands, styles, and tests.

### Stack Repository

Owns:

- version-pinned inference and backend container builds;
- Docker Compose networking and startup ordering;
- health checks and host port publication;
- read-only runtime-artifact mounting;
- local-stack verification scripts;
- clean-clone startup documentation.

The stack repository consumes published source refs. It does not duplicate application source code or include the native WPF application in a Linux container.

## Model Repository Structure

```text
configs/
  splits/
docs/
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
      app.py
      heatmap_encoding.py
      prediction_response.py
      prediction_routes.py
      runtime.py
      settings.py
    evaluation.py
    inference.py
    preprocessing.py
    training.py
    visualization.py
tests/
```

Virtual environments, datasets, reports, caches, ONNX exports, model artifacts, heatmaps, and experiment outputs are excluded from version control.

## Python Model Architecture

### General Fitting Flow

```text
normal image directory
-> discover_image_paths
-> create_image_path_split
-> ImagePathSplit
-> train_model_artifact
-> ImagePathDataset + preprocessing
-> ResNet18 patch embedding extractor
-> fitting feature memory
-> validation scores and threshold
-> metadata.json + feature_memory.pt
-> training_split.json
```

The general fitting boundary accepts explicit fitting and validation path tuples. It does not know MVTec folder conventions. Dataset-specific CLIs resolve their inputs and delegate to the same reusable training function.

### Image Discovery

`discover_image_paths` recursively discovers regular files with `.png`, `.jpg`, or `.jpeg` suffixes, case-insensitively. It returns resolved paths in deterministic order and rejects missing directories or directories without supported images.

### Deterministic Splitting

`create_image_path_split`:

- requires at least two unique paths;
- accepts a validation fraction strictly between zero and one;
- shuffles deterministically with an explicit seed;
- keeps fitting and validation non-empty;
- sorts both output partitions;
- verifies complete coverage and no overlap.

The generalized exporter writes `training_split.json` after successful artifact export. Paths are relative to the supplied image directory, which avoids machine-specific absolute paths.

### MVTec Manifest Compatibility

Versioned manifests remain available for established MVTec experiments:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
configs/splits/mvtec-ad-capsule-seed-42.json
```

Manifest loading validates counts, duplicates, overlap, absolute paths, parent traversal, and dataset-root resolution. The refactored MVTec exporter delegates to the same `train_model_artifact` implementation as the generalized directory exporter.

### Preprocessing

The selected pipeline is:

```text
image decode
-> RGB conversion
-> direct resize to configured square input
-> tensor conversion
-> ImageNet normalization
```

Direct resizing was selected because center cropping removed relevant Bottle boundaries. The artifact stores input size, while detailed interpolation, antialiasing, RGB conversion, mean, and standard deviation currently remain defined by the versioned Python implementation.

### Feature Extraction and Embeddings

A pretrained ResNet18 runs in evaluation mode with gradients disabled. Forward hooks capture `layer2` and `layer3`. The lower-resolution `layer3` features are resized to the `layer2` grid and concatenated, producing 384-dimensional patch embeddings.

| Input | Patch grid | Embeddings per image |
| --- | --- | ---: |
| 224 x 224 | 28 x 28 | 784 |
| 320 x 320 | 40 x 40 | 1,600 |

### Feature Memory

The feature memory contains embeddings from normal fitting images and is fitted category-specific state. It must not be mixed with incompatible preprocessing, backbone weights, feature layers, dimensions, or input resolutions.

Exact memory can be sampled deterministically, but experiments showed substantial Capsule recall loss. Complete memory remains the quality baseline.

### Scoring and Threshold

Each query patch is compared with its nearest fitting-memory embedding through exact chunked Euclidean distance calculation. Chunking limits temporary allocations but does not reduce the stored feature memory.

Patch scores are reshaped to the configured grid. The selected image score is the mean of the highest-scoring one percent of patches. The threshold is the maximum score among normal validation images. Scores strictly above the threshold are anomalous.

Test labels, defect folders, and masks are excluded from fitting and threshold selection.

### Visualization

Patch-score grids can be resized, normalized, colorized, and blended with source images. The service uses threshold-based normalization, generates an RGB image at model-input resolution, encodes it as PNG, and transports it as Base64 text.

Heatmaps are explanatory aids. They do not establish localization accuracy without quantitative comparison against masks.

## Training Orchestration

`ModelTrainingConfiguration` centralizes:

- batch size;
- nearest-neighbor memory chunk size;
- input size;
- top-score fraction;
- feature-memory sampling fraction;
- sampling seed.

`train_model_artifact` owns dataset construction, preprocessing, extractor creation, feature-memory fitting, validation scoring, threshold calculation, metadata construction, artifact persistence, and timing results.

CLI scripts own argument parsing, dataset-specific path resolution, automatic split creation, split-manifest persistence, and human-readable output. This prevents training logic from being duplicated across entry points.

## Model Artifact Architecture

### Core Artifact

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

Metadata contains:

- schema version;
- dataset and category;
- backbone;
- input and patch-grid sizes;
- embedding dimension;
- aggregation method and fraction;
- threshold;
- memory fraction and sampling seed;
- feature-memory entry count.

The writer validates tensor dimensions, finiteness, entry count, and embedding dimension. The loader reconstructs typed metadata and loads the tensor on CPU with `weights_only=True`.

### Generalized Split Sidecar

The generalized exporter additionally creates:

```text
training_split.json
```

It contains dataset and category identity, seed, fitting and validation ratios, counts, and exact relative path membership. It records artifact provenance without embedding local absolute paths.

### Verified Compatibility

Refactoring the Capsule manifest exporter through the shared training orchestrator reproduced identical metadata and the exact established feature-memory SHA-256 hash:

```text
51DE3F2B4FEF804E9E95900597E738E86F7044A669D2739956CBA0CC6DE65478
```

### Future Artifact Evolution

Future schema evolution may add:

- complete preprocessing semantics;
- pretrained weight identity;
- source-code and dependency versions;
- artifact and tensor checksums;
- training split checksum or embedded provenance reference;
- evaluation context and metrics;
- license and attribution information.

The current PyTorch tensor format is a trusted deployment input, not an untrusted upload format.

## Python Inference APIs

File-path and binary-stream APIs share the same pipeline:

```text
decode image
-> artifact-defined preprocessing
-> patch embeddings
-> exact nearest-neighbor scores
-> image aggregation
-> threshold comparison
-> score, decision, and patch grid
```

The stream boundary supports HTTP multipart uploads without coupling inference to temporary filesystem paths.

## Internal FastAPI Service

### Runtime Lifecycle

At startup:

1. settings read `IVAD_MODEL_ARTIFACT` and `IVAD_MEMORY_CHUNK_SIZE`;
2. configuration is validated;
3. the artifact is loaded once;
4. the frozen extractor is created once;
5. both are stored in an `InferenceRuntime`;
6. subsequent requests reuse the runtime.

Endpoints:

```text
GET  /health/live
POST /api/v1/predictions
```

The prediction endpoint accepts multipart field `image` and returns model identity, score, threshold, anomaly decision, and a threshold-normalized heatmap.

Prediction execution is guarded by a process-local lock. This is safe for the shared runtime but limits parallel throughput. Multiple workers would duplicate the large feature memory in RAM.

## ASP.NET Core Boundary

The backend is the public trust and compatibility boundary. It owns upload limits, file-signature validation, public Problem Details, trace identifiers, readiness behavior, cancellation, and mapping of internal service responses.

```text
client multipart request
-> backend validation
-> application anomaly-analyzer abstraction
-> HTTP adapter
-> FastAPI prediction
-> client-neutral backend response
```

The backend contract does not expose Python types or UI-specific presentation instructions.

## WPF Desktop Boundary

The desktop client calls only the backend. It owns image selection, preview, commands, busy state, result presentation, and heatmap interaction.

The verified UI overlays the heatmap on the source image with matching stretch behavior. Users can toggle visibility and adjust opacity. Decision colors remain presentation concerns derived from the backend decision.

## Docker Compose Architecture

The stack repository builds the inference and backend images from versioned Git tags. Compose provides:

- an internal application network;
- service-name resolution from backend to inference;
- inference health checks;
- backend startup after healthy inference;
- backend and inference liveness checks;
- host port publication;
- read-only artifact mounting;
- environment-based runtime configuration.

The ResNet18 weights are embedded during image build so inference startup does not require network access. The fitted feature memory remains a local runtime artifact rather than part of the container image.

## Configuration and Persistence

Reproducibility-critical model behavior belongs in artifact metadata or the split sidecar. Deployment-specific values such as ports, artifact paths, service addresses, timeouts, and memory chunk sizes belong in configuration.

Datasets, uploaded images, generated artifacts, and heatmaps are not persisted by the current backend or desktop client.

## Security and Safety

- the backend applies public upload validation and limits;
- the Python service validates malformed or unreadable images defensively;
- artifact paths come from controlled configuration;
- artifacts are trusted local inputs;
- generated artifacts and datasets remain outside Git;
- the system is experimental and not certified for production inspection decisions.

Authentication, encrypted service-to-service transport, secret management, hardened network isolation, signed artifacts, and production operational controls remain future work.

## Testing Strategy

### Python Tests

The current 92 tests cover:

- preprocessing, embeddings, memory, scoring, and evaluation;
- artifact metadata, persistence, and invalid inputs;
- path and stream inference;
- image discovery and deterministic splitting;
- split-manifest writing and path safety;
- training configuration and orchestration invariants;
- service settings and runtime lifecycle;
- liveness, multipart prediction, validation, and heatmap encoding.

Large artifacts, licensed datasets, and full cross-process integration are excluded from CI and verified manually.

### Backend and Desktop Tests

The backend separately tests health, Problem Details, configuration, upload validation, service mapping, heatmap contracts, and failure behavior. The desktop separately tests configuration, health communication, analysis mapping, image loading, heatmap decoding, and view-model behavior.

### Stack CI

The stack CI validates Compose configuration, parses the PowerShell verification script, and builds both Linux images. Real startup requires a local artifact and remains a documented manual clean-clone verification.

## Known Architectural Risks

- exact nearest-neighbor search is memory and compute intensive;
- multiple inference workers duplicate feature memory;
- a maximum-normal threshold may be sensitive to small validation sets;
- direct square resizing may distort non-square products;
- one-artifact runtime configuration does not yet support automatic category selection;
- PyTorch tensor artifacts require trusted inputs;
- Base64 heatmaps increase internal response size;
- service and backend contracts require coordinated versioning;
- dataset licenses constrain redistribution and commercial use;
- explanatory heatmaps can be overinterpreted without localization metrics.

## Current Non-Goals

- supervised defect classification;
- online learning during inference;
- automatic selection among multiple artifacts;
- public exposure of the FastAPI service;
- untrusted artifact uploads;
- certified production inspection;
- database persistence;
- a web client in the current milestone.

## Completed Architectural Milestones

1. Dataset validation and machine-readable reports.
2. Deterministic Bottle and Capsule manifests.
3. Reusable preprocessing, feature, scoring, evaluation, and visualization modules.
4. Complete and sampled feature-memory support.
5. Typed artifact metadata, writer, loader, and inference.
6. Capsule reference artifact and exploratory evaluation.
7. FastAPI runtime with persistent artifact loading.
8. Threshold-normalized PNG heatmap response.
9. ASP.NET Core public integration boundary.
10. WPF desktop analysis and interactive heatmap overlay.
11. Docker Compose backend and inference orchestration.
12. General PNG/JPEG directory discovery.
13. Deterministic general fitting and validation split.
14. Portable `training_split.json` persistence.
15. Shared dataset-independent training orchestrator.
16. Generalized 320 x 320 Bottle artifact verification.
17. Byte-for-byte Capsule exporter compatibility verification.
18. Ninety-two passing Python tests.

## Immediate Architectural Steps

1. Evaluate the generalized workflow on a genuinely non-MVTec image collection.
2. Define the external dataset contract and minimum useful normal-image counts.
3. Add artifact evaluation against optional normal and anomalous test directories.
4. Design explicit multi-artifact selection across service, backend, and clients.
5. Strengthen preprocessing provenance and artifact integrity metadata.
6. Investigate coverage-preserving memory reduction and faster search.
7. Add lightweight fixed-fixture cross-service contract coverage where practical.

## Related Documentation

- `DevelopmentStatus.md` records verified progress and current next steps.
- `DatasetDocumentation.md` records dataset provenance, storage, and licensing.
- `ModelDevelopmentStrategy.md` defines fitting, validation, and experiment rules.
- `ProjectSpecification.md` defines scope and requirements.
- a future `ModelCard.md` should document a released evaluated artifact.

## Last Updated

2026-08-19
