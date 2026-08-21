# Industrial Visual Anomaly Detection – Architecture Overview

## Purpose

This document describes the implemented architecture, repository boundaries, model-fitting workflow, artifact and registry contracts, inference-service integration, and current evolution path of the Industrial Visual Anomaly Detection system.

The system consists of independently versioned repositories for:

- Python model development and inference;
- the ASP.NET Core public backend;
- the native WPF desktop client;
- Docker Compose orchestration;
- a possible future web client.

## Architecture Status

### Implemented and Verified

- validation of MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- verified VisA Candle and Cashew ingestion and artifact workflows;
- recursive discovery of normal PNG and JPEG images;
- deterministic fitting and validation splits;
- portable split records with relative paths;
- frozen ResNet18 multi-scale feature extraction;
- 384-dimensional patch embeddings;
- complete and optionally sampled feature memories;
- exact chunked nearest-neighbor scoring;
- configurable normal-validation quantile thresholds;
- dataset-independent evaluation manifests and metrics;
- anomaly-map and heatmap generation;
- schema-versioned artifact export and compatibility loading;
- file-path and stream inference;
- legacy single-artifact service startup;
- validated multi-model registry configuration;
- startup loading of multiple enabled artifacts;
- default and explicit runtime selection;
- internal model-catalog and prediction endpoints;
- ASP.NET Core catalog and prediction integration;
- WPF runtime model selection and heatmap presentation;
- Docker Compose orchestration with a read-only registry and artifact mount;
- exploratory VisA Candle q95 threshold calibration;
- 144 automated Python test methods.

### Selected Direction

- Python remains authoritative for fitting, evaluation, artifact production, registry validation, and model inference;
- ASP.NET Core remains the public client-neutral API boundary;
- clients call the backend rather than the internal Python service;
- one fitted artifact represents one product category and model configuration;
- one registry describes the enabled deployment set and its default model;
- model selection uses stable identifiers rather than category-specific endpoints;
- normal-image directories are the general fitting input;
- MVTec manifests remain a reproducible specialized workflow;
- labeled CSV manifests provide the dataset-independent evaluation boundary;
- threshold quantiles remain category-specific calibration parameters in artifact metadata;
- HTTP remains the cross-runtime boundary;
- the WPF client remains native Windows software outside Docker;
- ONNX remains an optional portability path rather than a current integration requirement.

### Open

- independent validation of provisional threshold strategies;
- external dataset conventions and minimum useful image counts;
- aspect-ratio handling for non-square products;
- stronger artifact provenance and preprocessing metadata;
- checksums and artifact integrity policies;
- approximate nearest-neighbor search or coverage-preserving memory reduction;
- lazy model loading and controlled registry reload;
- quantitative pixel-level localization metrics;
- service authentication and production deployment hardening;
- a future web client.

## Architectural Goals

- keep datasets, registries, and generated artifacts outside source control;
- accept external normal-image collections without dataset-specific core logic;
- make discovery, splitting, fitting, thresholds, and artifacts reproducible;
- keep model logic independent from CLI, HTTP, and UI concerns;
- validate the configured deployment model set before serving requests;
- load large artifacts once per service process;
- select loaded models through stable identifiers;
- keep the public API independent of Python and UI implementation details;
- preserve CPU-only execution;
- prevent anomalous test images from influencing fitting or initial thresholds;
- separate exploratory calibration from independent final evaluation;
- evolve artifact, registry, and HTTP contracts explicitly.

## System Context

```text
External normal-image collections
        |
        v
Python fitting and artifact export
        |
        v
Category-specific model artifacts
        |
        +--> models.json registry
        |        |
        |        v
        +--> Internal FastAPI inference service
                  |
                  v
              ASP.NET Core public backend
                  |
                  +--> WPF desktop client
                  |
                  `--> future web client

Docker Compose stack
  -> builds and runs FastAPI + ASP.NET Core
  -> mounts registry and artifact directories read-only
  -> publishes backend and optional inference ports
```

## Repository Boundaries

### Model Repository

Owns:

- dataset validation and image discovery;
- deterministic split generation and recording;
- preprocessing and feature extraction;
- feature-memory construction and sampling;
- anomaly scoring, threshold selection, and evaluation;
- artifact export, validation, loading, and inference;
- model-registry schema and validation;
- runtime loading and model selection;
- heatmap generation and encoding;
- the internal FastAPI service;
- model experiments and documentation.

### Backend Repository

Owns:

- the public HTTP API;
- upload and model-identifier validation;
- Problem Details and trace identifiers;
- liveness and readiness semantics;
- HTTP communication with Python;
- public model-catalog mapping;
- mapping internal prediction responses to client-neutral contracts;
- configuration, logging, timeouts, and failure behavior.

### Desktop Repository

Owns:

- native Windows presentation;
- backend health and readiness display;
- retrieval and presentation of the model catalog;
- selected-model state;
- image selection and preview;
- model-specific analysis submission and cancellation;
- result and interactive heatmap presentation;
- WPF-specific state, commands, styles, and tests.

### Stack Repository

Owns:

- inference and backend container builds from explicit source references;
- Docker Compose networking and startup ordering;
- health checks and host port publication;
- read-only registry and artifact mounting;
- local-stack verification scripts;
- clean-clone startup documentation.

The stack does not duplicate application source or place the native WPF application in a Linux container.

## Model Repository Structure

```text
configs/splits/
docs/experiments/
scripts/
src/industrial_visual_anomaly_detection/
  artifacts/
  datasets/
  models/
  service/
    app.py
    heatmap_encoding.py
    model_registry_config.py
    model_routes.py
    prediction_response.py
    prediction_routes.py
    runtime.py
    runtime_registry.py
    settings.py
  evaluation.py
  inference.py
  preprocessing.py
  training.py
  visualization.py
tests/
```

Virtual environments, datasets, reports, caches, ONNX exports, registries, model artifacts, heatmaps, and experiment outputs are excluded from version control.

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
-> validation scores and quantile threshold
-> metadata.json + feature_memory.pt
-> training_split.json
```

The general fitting boundary accepts explicit fitting and validation paths and does not know MVTec folder conventions. Dataset-specific CLIs resolve inputs and delegate to the reusable training function.

### Image Discovery and Splitting

`discover_image_paths` recursively finds `.png`, `.jpg`, and `.jpeg` files case-insensitively in deterministic order.

`create_image_path_split` requires at least two unique paths, validates the fraction, shuffles with an explicit seed, keeps both partitions non-empty, sorts them, and verifies complete non-overlapping coverage.

The generalized exporter writes relative paths to `training_split.json`, avoiding machine-specific absolute paths.

### Dataset-Independent Evaluation

Labeled CSV manifests use `image`, `group`, and `is_anomalous`. Paths resolve relative to an explicit dataset root. Loading rejects invalid labels, duplicates, missing or unsupported files, absolute paths, and traversal outside that root.

The evaluator loads one artifact once and reports distributions, confusion-matrix counts, classification metrics, group anomaly rates, false positives, and false negatives.

### Preprocessing

```text
image decode
-> RGB conversion
-> direct resize to configured square input
-> tensor conversion
-> ImageNet normalization
```

Direct resizing avoids removing product boundaries through center cropping. Detailed interpolation, antialiasing, RGB conversion, mean, and standard deviation currently remain defined by the versioned implementation.

### Feature Extraction and Memory

A pretrained ResNet18 runs in evaluation mode with gradients disabled. Hooks capture `layer2` and `layer3`; resized `layer3` features are concatenated with `layer2`, producing 384-dimensional patch embeddings.

| Input | Patch grid | Embeddings per image |
| --- | --- | ---: |
| 224 × 224 | 28 × 28 | 784 |
| 320 × 320 | 40 × 40 | 1,600 |

Feature memory is category-specific fitted state. It must not be mixed with incompatible preprocessing, backbone weights, layers, dimensions, or input resolutions. Deterministic sampling is supported, while complete memory remains the quality baseline.

### Scoring and Thresholds

Each query patch is compared with its nearest fitting-memory embedding through exact chunked Euclidean distance. The selected image score is the mean of the highest-scoring one percent of patches.

The anomaly threshold is selected from normal validation scores through a configurable quantile. Quantile `1.0` preserves maximum-normal behavior. Lower quantiles may improve recall while increasing false positives. Threshold method and quantile are stored in artifact metadata.

Test labels, defect folders, and masks are excluded from fitting and initial threshold calculation. Later comparison using labeled test results is exploratory calibration, not independent final evaluation.

## Model Artifact Architecture

### Core Artifact

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

Metadata contains schema version, dataset, category, backbone, input and grid sizes, embedding dimension, aggregation, threshold method and quantile, sampling configuration, and memory entry count.

The writer validates dimensions, finiteness, entry count, and embedding dimension. New generalized artifacts use schema version 2. The loader supplies maximum-normal compatibility defaults for schema-version-1 metadata and loads tensors on CPU with `weights_only=True`.

### Generalized Split Sidecar

`training_split.json` records dataset and category identity, split seed, ratios, counts, and relative fitting and validation membership without local absolute paths.

### Artifact Trust Boundary

The current PyTorch tensor format is trusted local deployment input, not an untrusted upload format. Future schema evolution may add complete preprocessing semantics, weight identity, dependency versions, checksums, provenance, evaluation context, and licensing data.

## Model Registry Architecture

### Registry Layout

The deployment registry is a JSON document stored beside or above its artifact directories:

```text
artifact-root/
  models.json
  mvtec-ad-capsule-320/
    metadata.json
    feature_memory.pt
  visa-cashew-generalized-q95-320/
    metadata.json
    feature_memory.pt
```

The schema contains:

- `schemaVersion`;
- `defaultModelId`;
- an ordered `models` array;
- stable model `id`;
- human-readable `displayName`;
- relative `artifactDirectory`;
- `enabled` state.

### Registry Validation

`ModelRegistryConfiguration` validates the complete document before runtime loading. Validation includes:

- supported schema and expected fields;
- non-empty model collection;
- unique model identifiers;
- unique artifact directories;
- safe model-identifier format;
- relative artifact paths without parent traversal;
- at least one enabled model;
- an enabled default model;
- existence of every enabled artifact directory.

Disabled entries may reference artifacts not present locally because they are excluded from runtime loading and the public catalog.

### Runtime Registry

`InferenceRuntimeRegistry` loads one `InferenceRuntime` for each enabled registry entry. Every runtime owns its artifact, feature extractor, model identity, and request lock.

The registry:

- preserves configured model order;
- exposes runtime-derived category and input size;
- identifies the default model;
- resolves an explicit model identifier;
- uses the default when no identifier is supplied;
- rejects empty and unknown identifiers clearly.

All enabled models are loaded during startup. Dynamic reload and lazy loading are not currently implemented.

## Internal FastAPI Service

### Configuration Modes

The service supports two mutually exclusive sources:

```text
IVAD_MODEL_REGISTRY
IVAD_MODEL_ARTIFACT
```

Registry mode is the multi-model deployment path. `IVAD_MODEL_ARTIFACT` preserves compatibility with the earlier single-artifact workflow. Exactly one must be configured. `IVAD_MEMORY_CHUNK_SIZE` applies to runtime nearest-neighbor processing.

### Runtime Lifecycle

At startup:

1. settings validate that exactly one model source is configured;
2. registry mode loads and validates `models.json`;
3. each enabled artifact and extractor is loaded into an `InferenceRuntime`;
4. the runtimes are collected in `InferenceRuntimeRegistry`;
5. legacy mode loads one compatible `InferenceRuntime`;
6. the selected runtime source is stored in application state;
7. subsequent requests reuse the loaded state.

### Endpoints

```text
GET  /health/live
GET  /api/v1/models
POST /api/v1/predictions
```

`GET /api/v1/models` exposes the default and enabled models. Legacy mode is represented as a one-model catalog.

`POST /api/v1/predictions` accepts multipart field `image` and optional `modelId`. Registry mode resolves the requested runtime or falls back to the default. The response returns the actual model identifier, category, score, threshold, anomaly decision, and threshold-normalized heatmap.

Prediction execution is guarded by each runtime's process-local lock. This protects shared runtime state but limits parallel throughput per model. Multiple process workers would duplicate every loaded feature memory in RAM.

## Integration Flows

### Catalog Flow

```text
models.json
-> FastAPI runtime registry
-> GET /api/v1/models
-> ASP.NET Core catalog provider
-> public GET /api/v1/models
-> WPF model selection
```

The inference registry is authoritative. Backend and clients map the catalog rather than maintaining separate model lists.

### Prediction Flow

```text
client image + modelId
-> backend validation
-> internal multipart image + modelId
-> FastAPI runtime selection
-> selected model inference
-> response with actual model identity
-> backend response
-> client result and heatmap
```

The backend owns public upload limits, file-signature checks, Problem Details, trace identifiers, readiness, cancellation, and compatibility mapping. The desktop owns selection state and presentation.

## Docker Compose Architecture

The stack builds inference and backend images from explicit source references. Compose provides:

- an internal application network;
- backend-to-inference service-name resolution;
- inference health checks;
- backend startup after healthy inference;
- host port publication;
- read-only mounting of the registry and artifact root;
- environment-based runtime configuration.

ResNet18 weights are embedded during image build so startup does not require network access. Registries and fitted memories remain external runtime inputs rather than image content.

## Configuration and Persistence

Reproducibility-critical model behavior belongs in artifact metadata or split sidecars. Deployment model membership and defaults belong in `models.json`. Ports, registry paths, service addresses, timeouts, and memory chunk sizes belong in deployment configuration.

Datasets, uploaded images, generated artifacts, and heatmaps are not persisted by the current backend or desktop client.

## Security and Safety

- the backend applies public upload validation and limits;
- Python validates malformed or unreadable images defensively;
- registry and artifact paths come from controlled configuration;
- traversal and unexpected registry fields are rejected;
- registry files and artifacts are trusted local inputs;
- generated data remains outside Git;
- read-only mounts protect containerized runtime inputs;
- the system is experimental and not certified for production inspection decisions.

Authentication, encrypted service-to-service transport, signed artifacts, secret management, hardened isolation, and production controls remain future work.

## Testing Strategy

The current 144 Python test methods cover:

- preprocessing, embeddings, memory, scoring, and evaluation;
- artifact metadata, persistence, and invalid inputs;
- path and stream inference;
- image discovery and deterministic splitting;
- manifest writing and path safety;
- threshold quantiles, metrics, and compatibility loading;
- training configuration and orchestration;
- registry parsing, validation, ordering, paths, defaults, and disabled entries;
- multi-runtime loading and selection;
- legacy and registry service startup;
- liveness, catalog, multipart prediction, explicit selection, unknown models, validation, and heatmap encoding.

Large artifacts, licensed datasets, and full cross-process integration are excluded from Python CI and verified manually. Backend, desktop, and stack repositories maintain their own boundary tests.

## Known Architectural Risks

- exact nearest-neighbor search is memory and compute intensive;
- loading every enabled model increases startup time and resident memory;
- multiple process workers duplicate all feature memories;
- thresholds depend on representative validation data and operating trade-offs;
- test-informed calibration compromises independent final evaluation;
- direct square resizing may distort non-square products;
- PyTorch tensor artifacts require trusted inputs;
- the registry currently lacks artifact checksums;
- Base64 heatmaps increase response size;
- service, backend, and client contracts require coordinated versioning;
- dataset licenses constrain redistribution and commercial use;
- heatmaps can be overinterpreted without localization metrics.

## Current Non-Goals

- supervised defect classification;
- online learning during inference;
- automatic visual category recognition before model selection;
- dynamic registry editing or hot reload;
- public exposure of the FastAPI service;
- untrusted artifact uploads;
- certified production inspection;
- database persistence;
- a web client in the current milestone.

## Completed Architectural Milestones

1. Dataset validation and deterministic manifests.
2. Reusable preprocessing, feature, scoring, evaluation, and visualization modules.
3. Complete and sampled feature-memory support.
4. Typed artifact metadata, writer, loader, and inference.
5. FastAPI runtime with persistent artifact loading and heatmap responses.
6. ASP.NET Core and WPF MVP integration.
7. Docker Compose backend and inference orchestration.
8. General directory discovery and deterministic splitting.
9. Portable `training_split.json` persistence.
10. Dataset-independent training and evaluation workflows.
11. Schema-version-2 threshold metadata with schema-version-1 compatibility.
12. Exploratory VisA Candle q95 calibration.
13. Validated model-registry contract.
14. Startup loading of multiple enabled artifacts.
15. Default and explicit runtime selection.
16. Model-catalog endpoint and optional `modelId` prediction field.
17. Backend, desktop, and Compose multi-model integration.
18. One hundred forty-four automated Python test methods.

## Immediate Architectural Steps

1. Publish a registry-capable model-service release.
2. Validate threshold strategies on previously unused data.
3. Define a strict calibration and independent final-test protocol.
4. Strengthen preprocessing provenance and artifact integrity metadata.
5. Investigate coverage-preserving memory reduction and faster search.
6. Evaluate lazy model loading and controlled registry reload.
7. Add lightweight fixed-fixture cross-service contract coverage where practical.

## Related Documentation

- `DevelopmentStatus.md` records verified progress and next steps.
- `experiments/visa-candle-threshold-calibration.md` records the exploratory quantile comparison.
- `DatasetDocumentation.md` records dataset provenance, storage, and licensing.
- `ModelDevelopmentStrategy.md` defines fitting, validation, and experiment rules.
- `ProjectSpecification.md` defines stable scope and requirements.
- a future `ModelCard.md` should document a released evaluated artifact.

## Last Updated

2026-08-21
