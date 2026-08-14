# Industrial Visual Anomaly Detection – Architecture Overview

## Purpose

This document describes the implemented and intended architecture of the Industrial Visual Anomaly Detection system.

The system separates:

- external dataset storage and validation;
- Python model development, evaluation, artifact export, and inference execution;
- an internal FastAPI inference service;
- a separate client-neutral ASP.NET Core backend;
- future web and desktop clients.

The Python model pipeline, artifact format, CLI inference, internal HTTP inference service, and first backend-to-model request path are implemented. Web and desktop clients remain future components.

## Architecture Status

This document uses three status categories:

- **Implemented and verified** – demonstrated through executable code, automated tests, or recorded experiments;
- **Selected direction** – established as the current target but not yet complete;
- **Open** – requires further design, implementation, or evaluation.

### Implemented and Verified

- Python 3.12 CPU-based model-development and inference environment;
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
- model-artifact export, validation, and loading;
- file-path and binary-stream inference;
- single-image prediction CLI;
- internal FastAPI service with startup-time artifact loading;
- liveness and multipart prediction endpoints;
- serialized access to the shared inference runtime;
- separate ASP.NET Core backend with an HTTP adapter for the Python service;
- successful local end-to-end prediction through ASP.NET Core and FastAPI;
- provisional ONNX feature-extractor export and earlier PyTorch/ONNX parity work;
- 65 passing automated tests in the Python repository.

### Selected Direction

- the current reference model is MVTec AD Capsule at 320 x 320;
- the reference configuration uses complete feature memory and top-one-percent mean aggregation;
- Python remains the authoritative runtime for fitting, evaluation, artifact production, and model inference;
- a long-running FastAPI process provides the internal inference boundary;
- ASP.NET Core owns the public application API, validation, error mapping, and client-neutral contracts;
- React and WPF clients should consume the ASP.NET Core backend rather than call Python directly;
- model, backend, web, and desktop concerns remain independently versioned in separate repositories;
- ONNX remains an optional portability and optimization path rather than a prerequisite for backend integration.

### Open

- service readiness behavior and structured internal error responses;
- additional defense-in-depth validation inside the Python service;
- cancellation, timeout, concurrency, and throughput policies;
- service packaging, deployment, and process supervision;
- authentication or network isolation for the internal service boundary;
- framework-neutral feature-memory serialization;
- complete preprocessing metadata in the artifact contract;
- checksums and stronger artifact integrity validation;
- approximate nearest-neighbor or coverage-preserving coreset optimization;
- pixel-level metrics and anomaly-map post-processing;
- backend persistence and client implementation.

## Architectural Goals

The architecture should:

- keep model research independent from presentation technology;
- keep datasets and generated artifacts outside source control;
- make splits, fitting, thresholds, and experiments reproducible;
- preserve preprocessing and scoring semantics in the authoritative Python runtime;
- avoid loading the model artifact for every request;
- support image-level detection and patch-level localization;
- expose stable, client-neutral results through ASP.NET Core;
- keep CPU inference viable;
- make artifacts versioned, traceable, and compatible with their runtime;
- prevent test data from influencing fitting or threshold selection;
- support future model replacement without rewriting clients;
- isolate Python model failures from public API contracts;
- distinguish an experimental benchmark system from a validated industrial inspection system.

## System Context

```text
External datasets
        |
        v
Python model-development pipeline
        |
        v
Evaluated Python/PyTorch artifact
        |
        v
Internal FastAPI inference service
        |
        v
ASP.NET Core backend
        |
        v
Client-neutral public API
        |
        +--> future web client
        `--> future desktop client
```

The Python repository owns model behavior. The backend repository owns application-facing HTTP behavior. Clients do not duplicate model preprocessing, scoring, or threshold rules.

## Repository Boundaries

The intended repository separation is:

```text
industrial-visual-anomaly-detection-model
industrial-visual-anomaly-detection-backend
industrial-visual-anomaly-detection-web
industrial-visual-anomaly-detection-desktop
```

The model and backend repositories exist. Web and desktop repositories remain planned.

### Model Repository

The model repository owns:

- dataset documentation and validation;
- deterministic split creation;
- preprocessing experiments and implementation;
- feature extraction and patch embeddings;
- feature-memory creation and sampling experiments;
- anomaly scoring, threshold selection, and evaluation;
- heatmap generation;
- ONNX feasibility experiments;
- artifact export, loading, validation, and reference inference;
- the internal FastAPI service and its model-runtime lifecycle;
- the internal prediction response expected by the backend;
- experiment and artifact metadata design.

It does not own public client contracts, ASP.NET Core hosting, UI presentation, authentication, or application persistence.

### Backend Repository

The backend repository owns:

- public upload validation and request-size policy;
- versioned client-neutral analysis endpoints;
- application-level inference abstraction;
- the HTTP adapter to the internal Python service;
- configuration of the Python service address and timeout;
- mapping internal failures to stable Problem Details responses;
- trace identifiers and processing-time reporting;
- liveness and readiness endpoints;
- future persistence, authentication, observability, and deployment concerns.

The backend does not reproduce PyTorch preprocessing, embeddings, nearest-neighbor search, or threshold logic.

### Future Web and Desktop Repositories

Clients should own image selection, previews, request submission, result visualization, history views, platform-specific interaction, and presentation tests. They must call the backend rather than the internal Python service.

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
    service/
      app.py
      prediction_response.py
      prediction_routes.py
      runtime.py
      settings.py
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

Virtual environments, datasets, reports, caches, ONNX exports, model artifacts, heatmaps, and experiment outputs are excluded from version control.

## Python Model Architecture

```text
validated normal images
        |
        v
deterministic fitting and validation manifest
        |
        v
category configuration and preprocessing
        |
        v
frozen pretrained ResNet18
        |
        v
layer2 and layer3 feature maps
        |
        v
384-dimensional patch embeddings
        |
        v
complete or sampled normal feature memory
        |
        v
exact chunked nearest-neighbor distances
        |
        v
patch-score grid
        |
        v
image-score aggregation
        |
        v
normal-validation threshold
        |
        v
decision, evaluation, heatmap, and artifact export
```

### Dataset Boundary

Datasets are stored outside the repository. Runtime commands receive the local root explicitly, and public configuration does not contain machine-specific dataset roots.

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

Bottle uses 224 x 224. The Capsule reference uses 320 x 320. Direct resizing was selected for Bottle because center cropping removed object-boundary information.

The artifact currently stores the input size. Mean, standard deviation, interpolation, antialiasing, and RGB conversion remain defined by the versioned Python implementation. A more portable artifact must store these semantics explicitly.

### Feature Extractor and Patch Embeddings

For 224 x 224 input:

```text
layer2:          (1, 128, 28, 28)
layer3:          (1, 256, 14, 14)
patch embeddings:      (784, 384)
```

For 320 x 320 input:

```text
layer2:          (1, 128, 40, 40)
layer3:          (1, 256, 20, 20)
patch embeddings:     (1600, 384)
```

`layer3` is resized to the spatial resolution of `layer2`, concatenated along the channel dimension, and rearranged into one 384-dimensional embedding per patch position. The backbone is frozen and kept in evaluation mode.

### Feature Memory

The feature memory contains normal fitting embeddings and is part of the fitted model state. It must never be combined with incompatible preprocessing, backbone weights, feature layers, dimensions, or input resolution.

The Capsule artifact contains:

```text
shape: (280000, 384)
dtype: float32
size:  approximately 410.16 MiB
```

Deterministic random sampling at 75%, 50%, and 25% reduced runtime but degraded recall. Complete memory remains the reference configuration. Smarter coreset selection remains open.

### Nearest-Neighbor Scoring

For every query patch, the implementation computes its Euclidean distance to the nearest normal feature-memory entry. Memory is processed in configurable chunks to bound temporary distance tensors while preserving exact results.

Distances are reshaped into a patch grid. Supported image aggregation includes the maximum patch score and the mean of the highest configurable fraction. Top-one-percent mean is selected for the Capsule artifact.

### Threshold and Evaluation

The threshold is selected exclusively from normal validation images as their maximum image-level score. Only scores strictly above the threshold are anomalous.

Official test labels and masks are not used for fitting or threshold calculation. Test-folder names support grouped analysis. Current metrics include confusion-matrix counts, accuracy, precision, recall, F1, per-group detection rates, and runtime measurements.

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

Typed metadata contains:

- schema version;
- dataset and category;
- backbone name;
- input and patch-grid sizes;
- embedding dimension;
- aggregation method and top fraction;
- validation-derived threshold;
- memory fraction and sampling seed;
- feature-memory entry count.

The writer validates shape, finite values, entry count, and embedding dimension. The loader verifies required files, reconstructs typed metadata, loads the tensor on CPU with `weights_only=True`, and validates tensor type, dimensionality, finite values, entry count, and embedding dimension.

The local format is sufficient for trusted Python inference but is not framework-neutral. Artifacts are generated locally, ignored by Git, and must not be accepted from untrusted upload sources.

### Reference Capsule Artifact

```text
dataset:                mvtec-ad
category:               capsule
backbone:               resnet18
input size:             320 x 320
patch grid:             40 x 40
embedding dimension:    384
feature-memory entries: 280000
aggregation:            top_fraction_mean
top fraction:           0.01
memory fraction:        1.0
threshold:              2.501821517944336
```

### Future Artifact Evolution

A future portable package may include:

```text
model-package/
  feature-extractor.onnx
  feature-memory.<portable-format>
  model-metadata.json
  evaluation-summary.json
  checksums.txt
  notices/
```

This is an optional portability path, not the current backend integration requirement. Metadata should eventually identify the pretrained weight version, feature layers, full preprocessing semantics, split manifest, software versions, artifact version, license attribution, and evaluation context.

## Python Inference APIs

Two inference entry points share the same model behavior:

- `predict_image` accepts a filesystem path and supports CLI workflows;
- `predict_image_stream` accepts a binary stream and supports HTTP uploads without temporary path coupling.

Both return scores generated by the same preprocessing, extractor, feature memory, aggregation, and stored threshold. Path and stream inference were verified to produce identical score, threshold, and decision for the same Capsule image.

## Internal FastAPI Service

### Runtime Lifecycle

At application startup:

1. `InferenceServiceSettings` reads and validates environment configuration;
2. `InferenceRuntime.load` loads the artifact once;
3. the frozen ResNet18 patch-embedding extractor is created once;
4. the runtime is stored in FastAPI application state;
5. subsequent requests reuse the loaded artifact and extractor.

The current environment variables are:

```text
IVAD_MODEL_ARTIFACT
IVAD_MEMORY_CHUNK_SIZE
```

The artifact path is required. The memory chunk size defaults to `4096` and must be a positive integer.

### Concurrency Policy

`InferenceRuntime` protects prediction execution with a process-local lock. This conservative first implementation prevents concurrent use of the shared extractor and large feature memory. It prioritizes correctness and predictable resource use over throughput.

Concurrency benchmarking and a deliberate worker policy are required before increasing parallelism. Multiple Uvicorn workers would independently load the approximately 410 MiB feature memory and therefore multiply memory consumption.

### Internal HTTP Contract

Liveness:

```text
GET /health/live
```

Prediction:

```text
POST /api/v1/predictions
Content-Type: multipart/form-data
Field: image
```

Example prediction response:

```json
{
  "modelId": "mvtec-ad-capsule-320",
  "category": "capsule",
  "score": 4.992109298706055,
  "threshold": 2.501821517944336,
  "isAnomalous": true
}
```

The artifact directory name currently acts as `modelId`. The endpoint is internal and does not replace the public backend contract.

## ASP.NET Core Integration

The implemented request path is:

```text
client uploads image
        |
        v
ASP.NET Core POST /api/v1/analyses
        |
        v
public upload validation and application orchestration
        |
        v
PythonServiceAnomalyAnalyzer
        |
        v
multipart HTTP request to FastAPI POST /api/v1/predictions
        |
        v
InferenceRuntime performs Python/PyTorch prediction
        |
        v
internal prediction response
        |
        v
backend maps result to client-neutral response
```

A verified local anomalous Capsule request returned:

```json
{
  "model": {
    "id": "mvtec-ad-capsule-320",
    "category": "capsule"
  },
  "score": 4.992109298706055,
  "threshold": 2.501821517944336,
  "decision": "anomalous",
  "processingTimeMs": 1802,
  "traceId": "<request-trace-id>"
}
```

The backend maps service unavailability and transport failures to its public error contract. Controllers depend on an application abstraction instead of directly depending on HTTP or Python details.

## Boundary Rationale

The Python-service approach was selected over embedding Python into the .NET process or immediately rewriting the complete pipeline in .NET because it:

- preserves the verified PyTorch implementation;
- avoids premature cross-runtime preprocessing and scoring parity work;
- keeps the large feature memory loaded in one long-running model process;
- isolates Python dependencies and failures from the public API process;
- provides a language-neutral HTTP boundary;
- allows backend and model repositories to evolve independently;
- leaves ONNX or native .NET inference available as later optimization options.

The trade-offs are an additional process, local HTTP overhead, service lifecycle management, and the need for timeouts and operational health checks.

## Client-Neutral Result Contract

The backend response may contain:

- model identity and category;
- anomaly score and threshold;
- normal/anomalous decision;
- processing duration;
- trace identifier;
- future anomaly-map or overlay references;
- stable validation and failure details.

Neither the backend nor the Python service should return UI-specific instructions such as colors, tabs, dialogs, WPF commands, or React component names.

## Configuration and Persistence

Configuration separates public defaults, local overrides, deployment values, and secrets. Reproducibility-critical model behavior belongs in artifact metadata rather than only environment configuration.

Optional future inspection persistence may record identifiers, timestamps, model versions, image references or checksums, scores, thresholds, decisions, durations, and failures. Raw image retention must be optional and governed by explicit privacy and retention rules.

## Security and Safety

The backend is the public trust boundary and must apply upload limits, content-type policy, stable errors, timeouts, and controlled forwarding. The Python service should still implement defense-in-depth validation for malformed or unsupported content.

Model artifacts are trusted deployment inputs because the current format uses PyTorch tensor serialization. Artifact paths must come from controlled configuration, not user input.

The project must not be represented as a validated production quality-control system without domain validation, controlled deployment, regulatory assessment where applicable, and operational monitoring.

## Testing Strategy

### Implemented Python Tests

The current 65 tests cover:

- manifests and dataset discovery;
- preprocessing;
- feature extraction and patch embeddings;
- feature-memory construction and sampling;
- nearest-neighbor distances;
- anomaly scoring and aggregation;
- evaluation metrics;
- visualization;
- artifact persistence;
- service-facing inference integration;
- service settings;
- runtime loading and reuse;
- FastAPI lifespan and liveness behavior;
- prediction response mapping and missing-upload validation.

Large artifacts and licensed datasets are not available in CI. Real artifact startup and the complete cross-process path are therefore verified locally rather than in the standard unit-test workflow.

### Backend Tests

The backend separately tests health endpoints, Problem Details, configuration binding, upload validation, analysis behavior, service response mapping, and service failure handling.

### Next Integration Coverage

Future tests should cover:

- unsupported or malformed images at the Python boundary;
- internal service error serialization;
- readiness before and after artifact loading;
- backend timeout and cancellation behavior;
- a lightweight backend-to-service contract fixture suitable for CI;
- artifact schema compatibility and corrupted metadata;
- concurrent request behavior;
- fixed-fixture regression scores for released artifacts.

## Observability and Deployment Direction

The backend should record trace identifiers, model identity, total request duration, service failures, and public outcomes without exposing sensitive image content.

The Python service should record startup state, artifact identity, prediction duration, failures, and resource use. Logs must not contain raw image content.

Initial deployment remains CPU-compatible and may run the backend and Python service as separately supervised local processes or containers. Production packaging, network restrictions, readiness probes, graceful shutdown, and artifact provisioning remain open.

## Known Architectural Risks

- complete feature memories are large;
- exact CPU nearest-neighbor search may limit throughput;
- multiple service workers duplicate the feature memory in RAM;
- the current serialized prediction lock limits concurrency;
- random sampling reduces quality on Capsule;
- small defects may require higher input resolution;
- non-square categories need a deliberate resize policy;
- threshold or hyperparameter selection can leak test information;
- service and backend contracts require coordinated versioning;
- process startup fails when the configured artifact is absent or incompatible;
- HTTP and process boundaries add deployment complexity;
- artifact evolution requires schema and compatibility management;
- dataset licenses constrain redistribution and commercial use;
- benchmark results may not transfer to real industrial imagery;
- a prototype may be mistaken for a validated inspection system.

## Current Non-Goals

- production deployment certification;
- real-time camera or PLC integration;
- a production database and authentication system;
- implemented web or desktop UI;
- supervised defect-type classification;
- automated retraining or multi-model orchestration;
- public exposure of the FastAPI service;
- untrusted artifact uploads;
- regulatory validation;
- real pharmaceutical packaging validation.

## Completed Architectural Milestones

1. Reproducible Python environment and dependency setup.
2. CPU-based pretrained ResNet18 and intermediate feature extraction.
3. Patch embeddings and ONNX feasibility experiments.
4. Acquisition and validation of three MVTec datasets.
5. Machine-readable dataset reports.
6. Deterministic Bottle and Capsule manifests.
7. Reusable dataset, preprocessing, model, scoring, evaluation, and visualization modules.
8. Bottle baseline and Capsule generalization evaluation.
9. Configurable resolution and aggregation experiments.
10. Feature-memory sampling implementation and trade-off evaluation.
11. Typed artifact metadata, writer, loader, and round-trip tests.
12. Export of the complete 320 x 320 Capsule reference artifact.
13. Verified normal and anomalous single-image CLI inference.
14. Binary-stream inference with parity against path-based inference.
15. FastAPI service with startup-time artifact and extractor initialization.
16. Internal liveness and multipart prediction endpoints.
17. ASP.NET Core HTTP adapter to the Python inference service.
18. Verified end-to-end anomalous Capsule request through both processes.

## Immediate Architectural Steps

1. Add Python-service readiness based on initialized runtime state.
2. Define structured internal errors and malformed-image behavior.
3. Add defense-in-depth content validation at the Python boundary.
4. Define backend timeout, cancellation, and retry policy explicitly.
5. Add service timing and structured logging.
6. Create a reproducible local startup workflow for both processes.
7. Add lightweight cross-service contract coverage where practical.
8. Continue the public backend analysis contract and error handling.
9. Evaluate deployment packaging and artifact provisioning.
10. Begin client work only after the public backend contract is sufficiently stable.

## Related Documentation

- `DevelopmentStatus.md` records verified results and active work.
- `ProjectSpecification.md` defines product scope and requirements.
- `ModelDevelopmentStrategy.md` defines fitting, validation, and experiment rules.
- `DatasetDocumentation.md` records sources, licenses, structures, and validation.
- A future `ModelCard.md` should document a released evaluated artifact.

## Last Updated

This architecture reflects the verified project state as of 2026-08-14.
