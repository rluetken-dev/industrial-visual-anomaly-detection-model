# Industrial Visual Anomaly Detection – Project Specification

## Document Purpose

This document defines the current functional and non-functional requirements for the Industrial Visual Anomaly Detection system.

It covers:

- the implemented Python model-development and inference pipeline;
- versioned Python/PyTorch model artifacts;
- the implemented internal FastAPI inference service;
- the implemented separate ASP.NET Core backend MVP;
- the implemented WPF desktop-client MVP;
- a planned web client;
- experimental, deferred, and production-hardening requirements.

Implementation history and immediate work are recorded in `DevelopmentStatus.md`. Architectural responsibilities and boundaries are described in `ArchitectureOverview.md`.

## Requirement Identification

Requirement identifiers use these prefixes:

- `FR` – Functional Requirement
- `BR` – Business Rule
- `NFR` – Non-Functional Requirement
- `AC` – Acceptance Criterion

Functional-area codes are:

- `DAT` – Dataset Management
- `PRE` – Image Preprocessing
- `MOD` – Model Development
- `INF` – Anomaly Inference
- `EVA` – Evaluation
- `VIS` – Visualization
- `ART` – Model Artifacts
- `SVC` – Internal Python Inference Service
- `API` – ASP.NET Core Backend API
- `WEB` – Web Application
- `DSK` – Desktop Application
- `OBS` – Observability and Diagnostics
- `SEC` – Security and Privacy

Identifiers follow `<type>-<area>-<number>`, for example `FR-INF-001`.

## Project Overview

Industrial Visual Anomaly Detection is a portfolio-oriented computer-vision system for detecting unusual visual patterns in industrial inspection images and highlighting suspicious regions.

The implemented model uses a PatchCore-inspired method:

1. normal images are preprocessed deterministically;
2. a frozen pretrained ResNet18 extracts intermediate features;
3. `layer2` and `layer3` features are combined into patch embeddings;
4. embeddings from normal fitting images form a feature memory;
5. query patches are compared with their nearest normal neighbor;
6. patch distances form an anomaly map;
7. an aggregation rule produces one image-level anomaly score;
8. a threshold derived only from held-out normal validation images determines the decision.

The system detects deviation from learned normal appearance. It does not classify the exact defect type.

Model inference remains authoritative in Python. A persistent internal FastAPI service exposes it to a separate ASP.NET Core backend, which owns the public application API and client integration. The implemented WPF desktop client consumes that backend contract.

## Current Implementation Status

### Implemented and Verified

- Python 3.12 CPU development and inference environment;
- reusable `src`-layout Python package;
- deterministic RGB preprocessing with direct square resize and ImageNet normalization;
- configurable input resolution;
- frozen pretrained ResNet18 `layer2` and `layer3` feature extraction;
- 384-dimensional multi-scale patch embeddings;
- complete normal feature-memory construction;
- deterministic random feature-memory sampling as an experimental option;
- exact chunked Euclidean nearest-neighbor search;
- patch-level anomaly maps;
- maximum and top-fraction-mean image-score aggregation;
- normal-validation-based threshold selection;
- MVTec AD category discovery and labeled test loading;
- image-level metrics and per-group reporting;
- anomaly heatmaps and overlays;
- versioned PyTorch model artifacts containing metadata and feature memory;
- artifact validation, writing, and loading;
- file-path and binary-stream inference;
- single-image prediction CLI;
- FastAPI service with startup-time artifact and extractor loading;
- FastAPI liveness and multipart prediction endpoints;
- threshold-normalized RGB heatmaps encoded as Base64 PNG data in prediction responses;
- serialized access to shared inference state;
- ASP.NET Core .NET 10 backend foundation;
- backend liveness, readiness, Problem Details, upload validation, and analysis endpoint;
- backend HTTP adapter to the Python inference service;
- manually verified end-to-end inference through both processes;
- WPF desktop client with health status, image selection, preview, submission, and result display;
- 68 passing Python tests plus automated backend and desktop tests;
- GitHub Actions CI in all three implemented repositories.

### Verified Datasets

Local copies of MVTec AD, MVTec LOCO AD, and MVTec AD 2 pass structural, inventory, image-readability, and available mask validations. Machine-readable JSON validation reports can be generated locally.

### Reference Categories

`bottle` was the first baseline category. Its 209 normal training images are split into 167 fitting and 42 validation images.

`capsule` is the current artifact and service reference category. Its 219 normal training images are split into 175 fitting and 44 validation images.

The selected Capsule configuration is:

- input size: 320 x 320;
- patch grid: 40 x 40;
- embedding dimension: 384;
- image aggregation: mean of the highest-scoring 1% of patches;
- threshold: maximum normal-validation score;
- feature memory: complete unsampled fitting memory;
- feature-memory shape: `(280000, 384)`;
- feature-memory size: approximately 410.16 MiB;
- stored threshold: `2.501821517944336`.

### Not Yet Implemented

- pixel-level benchmark metrics against ground-truth masks;
- systematic multi-category benchmark execution;
- complete preprocessing metadata and artifact checksums;
- Python-service readiness endpoint;
- structured internal service error contract;
- production timeout, cancellation, retry, and concurrency policies;
- service packaging and process supervision;
- backend persistence, authentication, and production observability;
- web application;
- production deployment and public model-artifact distribution.

## Project Goals

The project shall demonstrate:

- practical reuse of pretrained computer-vision features;
- normal-only anomaly fitting without defect examples;
- reproducible image-level anomaly detection;
- understandable spatial anomaly visualization;
- deterministic evaluation and artifact generation;
- reusable offline and stream-based inference;
- a persistent internal Python model service;
- a stable client-neutral ASP.NET Core API;
- independent model and application boundaries;
- transparent reporting of assumptions, limitations, and exploratory results.

## Intended Users

- developers learning computer vision and anomaly detection;
- engineers evaluating visual inspection workflows;
- portfolio reviewers examining architecture, testing, and reproducibility;
- WPF desktop users submitting inspection images and reviewing results;
- future web users using the same backend workflow.

The project is not certified for production quality control or safety-critical decisions.

## Scope

### Python Model and Service Scope

The current Python scope includes dataset qualification, deterministic splitting, preprocessing, feature-memory creation, anomaly scoring, evaluation, visualization, artifact export and loading, file-path and binary-stream inference, and the internal FastAPI service.

### Backend Scope

The ASP.NET Core backend owns the public image-analysis endpoint, upload validation, application orchestration, the Python-service adapter, stable error mapping, trace identifiers, and client access.

### Client Scope

The implemented WPF desktop client consumes the backend contract for health checks and image analysis. A future web client shall consume the same backend contract rather than duplicating model or business logic.

### Explicitly Deferred

- exact defect-type classification;
- supervised detector training;
- continuous video inspection;
- camera and programmable-logic-controller integration;
- automated production retraining;
- multi-tenant operation;
- safety certification;
- commercial dataset use without separate license review.

## Functional Requirements

### Dataset Management

- `FR-DAT-001` The system shall accept dataset roots outside the Git repository.
- `FR-DAT-002` Validators shall verify expected structure, inventory, readable images, and available mask relationships.
- `FR-DAT-003` Validators shall support optional schema-versioned JSON reports.
- `FR-DAT-004` Raw datasets and generated reports shall remain excluded from Git.
- `FR-DAT-005` A deterministic manifest shall identify fitting and normal-validation images.
- `FR-DAT-006` Manifest validation shall reject duplicate, overlapping, absolute, or parent-traversing paths.
- `FR-DAT-007` Resolved dataset paths shall be verified before model processing.
- `FR-DAT-008` MVTec AD test discovery shall preserve category, defect group, and normal/anomalous labels.

### Image Preprocessing

- `FR-PRE-001` Input images shall be converted to RGB.
- `FR-PRE-002` Images shall be resized directly to the configured square input using bilinear interpolation with antialiasing.
- `FR-PRE-003` Images shall become floating-point tensors normalized with ImageNet mean and standard deviation.
- `FR-PRE-004` Fitting, validation, evaluation, CLI, and service inference shall use the same preprocessing for a model configuration.
- `FR-PRE-005` The selected input size shall be recorded in artifact metadata.
- `FR-PRE-006` Inspection tooling shall permit visual verification of resize behavior.

### Model Development

- `FR-MOD-001` The reference baseline shall use pretrained ResNet18 with frozen parameters.
- `FR-MOD-002` The extractor shall expose `layer2` and `layer3` maps.
- `FR-MOD-003` The embedding pipeline shall resize `layer3` to the `layer2` grid and concatenate channels.
- `FR-MOD-004` Each patch embedding shall contain 384 values.
- `FR-MOD-005` Feature memory shall be built only from normal fitting images.
- `FR-MOD-006` Complete feature memory shall remain reference behavior.
- `FR-MOD-007` Optional memory sampling shall be deterministic for a fixed seed.
- `FR-MOD-008` Sampling experiments shall report retained fraction, memory size, runtime, and quality.

### Anomaly Inference

- `FR-INF-001` Each query patch shall use exact Euclidean distance to its nearest feature-memory entry.
- `FR-INF-002` Distance computation shall support chunking to limit temporary memory.
- `FR-INF-003` Patch scores shall be reconstructed into a spatial grid.
- `FR-INF-004` Image scores shall support maximum and top-fraction-mean aggregation.
- `FR-INF-005` A prediction shall be anomalous only when its score is strictly greater than the threshold.
- `FR-INF-006` Path-based inference shall return image path, score, threshold, decision, and patch-score grid.
- `FR-INF-007` Stream inference shall return score, threshold, decision, and patch-score grid without requiring a temporary path.
- `FR-INF-008` Inference shall derive input size, patch-grid size, aggregation, and threshold from the artifact.
- `FR-INF-009` CLI inference shall display model configuration, score, threshold, decision, and relevant timings.
- `FR-INF-010` Path and stream inference shall share the same preprocessing and scoring implementation.

### Evaluation

- `FR-EVA-001` The threshold shall derive only from held-out normal validation images.
- `FR-EVA-002` The current threshold rule shall use the maximum normal-validation score.
- `FR-EVA-003` Evaluation shall report validation and per-defect-group score distributions.
- `FR-EVA-004` Evaluation shall report confusion counts, accuracy, precision, recall, and F1.
- `FR-EVA-005` Evaluation shall report per-group detection and false negatives.
- `FR-EVA-006` Aggregation comparisons shall reuse the same patch scores where possible.
- `FR-EVA-007` Results influenced by inspected test data shall be labeled exploratory.
- `FR-EVA-008` Future pixel evaluation shall compare maps with masks using explicitly selected metrics.

### Visualization

- `FR-VIS-001` Patch-score grids shall resize to image resolution using bilinear interpolation.
- `FR-VIS-002` Maps shall support per-image normalization.
- `FR-VIS-003` Maps shall support threshold-based normalization for cross-image comparison.
- `FR-VIS-004` The system shall create RGB heatmaps and configurable-opacity overlays.
- `FR-VIS-005` Heatmaps shall remain explanation aids and not replace decision contracts.
- `FR-VIS-006` Internal service heatmaps shall use threshold-based normalization and shall be encoded as RGB PNG images.

### Model Artifacts

- `FR-ART-001` A current artifact shall contain `metadata.json` and `feature_memory.pt`.
- `FR-ART-002` Metadata shall include schema version, dataset, category, backbone, input and grid sizes, embedding dimension, aggregation, top fraction, threshold, memory fraction, sampling seed, and entry count.
- `FR-ART-003` The writer shall reject empty, non-finite, incorrectly shaped, or metadata-inconsistent memory.
- `FR-ART-004` The loader shall validate required files, typed metadata, tensor dimensions, finite values, entry count, and embedding dimension.
- `FR-ART-005` Artifact tensors shall load onto CPU with restricted tensor loading behavior.
- `FR-ART-006` Generated artifacts shall remain excluded from Git.
- `FR-ART-007` Export shall rebuild fitting memory, calculate the validation threshold, and write one reusable artifact.
- `FR-ART-008` The artifact format shall explicitly version its Python/PyTorch runtime dependency until a portable format is implemented.
- `FR-ART-009` Future metadata shall become fully self-describing for preprocessing and pretrained-weight identity.

### Internal Python Service

- `FR-SVC-001` The service shall read the artifact path from `IVAD_MODEL_ARTIFACT`.
- `FR-SVC-002` The service shall support a positive configurable memory chunk size through `IVAD_MEMORY_CHUNK_SIZE`.
- `FR-SVC-003` The artifact and feature extractor shall load once during application startup.
- `FR-SVC-004` Requests shall reuse the loaded runtime.
- `FR-SVC-005` `GET /health/live` shall report process liveness.
- `FR-SVC-006` `POST /api/v1/predictions` shall accept multipart field `image`.
- `FR-SVC-007` A successful response shall contain model ID, category, score, threshold, Boolean anomaly decision, and a nested heatmap.
- `FR-SVC-008` The artifact directory name shall currently identify the model.
- `FR-SVC-009` Shared runtime access shall remain serialized until measured concurrency behavior supports another policy.
- `FR-SVC-010` The service shall expose readiness only after runtime initialization is successful.
- `FR-SVC-011` The service shall eventually return structured internal errors for invalid images and inference failures.
- `FR-SVC-012` The internal endpoint shall not be treated as the public client API.
- `FR-SVC-013` The heatmap contract shall contain content type, width, height, and Base64-encoded PNG data.

### ASP.NET Core Backend API

- `FR-API-001` ASP.NET Core shall own public application-level inference orchestration.
- `FR-API-002` `POST /api/v1/analyses` shall accept an image and return a client-neutral anomaly result.
- `FR-API-003` The response shall include model identity, category, score, threshold, decision, processing duration, and trace identifier.
- `FR-API-004` The backend shall validate required upload, file size, and allowed PNG/JPEG media types.
- `FR-API-005` The backend shall depend on an anomaly-analyzer abstraction rather than Python-specific controller logic.
- `FR-API-006` The Python adapter shall forward multipart field `image` to the internal prediction endpoint.
- `FR-API-007` Service transport and availability failures shall map to stable Problem Details responses.
- `FR-API-008` The backend shall provide liveness and readiness endpoints.
- `FR-API-009` Web and desktop clients shall consume the same backend contract.
- `FR-API-010` The backend shall not silently change model artifacts, aggregation, or thresholds.

### Web Application – Planned

- `FR-WEB-001` The web client shall allow image selection and submission.
- `FR-WEB-002` It shall display decision, score, threshold, and model identity.
- `FR-WEB-003` It shall display localization output when supplied by the backend.
- `FR-WEB-004` Errors shall be understandable and omit sensitive server details.

### Desktop Application – Implemented MVP

- `FR-DSK-001` The desktop client shall allow local image selection and backend submission.
- `FR-DSK-002` It shall display the same essential result fields as the web client.
- `FR-DSK-003` Desktop convenience features shall not create a separate inference contract.
- `FR-DSK-004` The desktop client shall display backend liveness and inference-readiness state.
- `FR-DSK-005` The desktop client shall preview the selected image and display decision, score, threshold, model identity, category, duration, and trace identifier.
- `FR-DSK-006` The desktop client shall display localization output when supplied by the backend.

### Observability and Diagnostics

- `FR-OBS-001` Diagnostics shall record model identity, duration, outcome, and failure category without logging raw image content by default.
- `FR-OBS-002` Health information shall distinguish application liveness from model readiness.
- `FR-OBS-003` Backend logs and responses shall support request correlation identifiers.
- `FR-OBS-004` The Python service shall eventually record startup, artifact load, inference duration, and categorized failures.

## Business Rules

- `BR-MOD-001` Only normal fitting images may populate feature memory.
- `BR-EVA-001` Normal validation images may determine thresholds but may not populate feature memory.
- `BR-EVA-002` Test labels may support evaluation but not fitting or threshold selection.
- `BR-INF-001` A score equal to the threshold is normal; only a greater score is anomalous.
- `BR-ART-001` Each artifact is category-specific unless its schema explicitly declares otherwise.
- `BR-ART-002` A released artifact shall be immutable.
- `BR-SVC-001` The Python service is the authoritative model-execution runtime for the selected architecture.
- `BR-API-001` Clients shall call ASP.NET Core rather than the internal Python service.
- `BR-DAT-001` Dataset licensing shall be reviewed before public or commercial distribution.

## Non-Functional Requirements

### Reproducibility

- `NFR-MOD-001` Deterministic operations shall record seeds and configuration.
- `NFR-DAT-001` Versioned manifests shall make fitting and validation membership reproducible.
- `NFR-ART-001` Artifacts shall identify their model configuration.
- `NFR-EVA-001` Results shall state category, resolution, memory fraction, aggregation, and threshold rule.

### Performance and Resource Use

- `NFR-INF-001` The reference shall support CPU-only inference.
- `NFR-INF-002` Distance computation shall avoid the complete query-to-memory distance matrix.
- `NFR-INF-003` Measurements shall separate loading, extractor creation, memory construction, scoring, and end-to-end duration where relevant.
- `NFR-SVC-001` The service shall avoid loading the artifact for every request.
- `NFR-SVC-002` Worker and concurrency configuration shall account for duplicated feature-memory RAM.
- `NFR-SVC-003` Production latency and throughput targets shall be established through measurement rather than assumption.

### Compatibility and Portability

- `NFR-ART-002` Current artifacts shall be labeled Python/PyTorch artifacts rather than runtime-neutral artifacts.
- `NFR-API-001` Backend and service contracts shall be explicitly versioned when compatibility changes require coordination.
- `NFR-MOD-002` ONNX parity shall be reverified before ONNX is used in a supported runtime path.
- `NFR-SVC-004` Model-service dependency versions shall remain pinned and reproducible.

### Maintainability and Testability

- `NFR-MOD-003` Reusable model logic shall reside in the package rather than only in scripts.
- `NFR-MOD-004` Public package functions shall reject invalid or inconsistent inputs clearly.
- `NFR-MOD-005` Core deterministic components shall have automated tests.
- `NFR-MOD-006` Scripts shall compose reusable package components.
- `NFR-SVC-005` FastAPI construction shall allow injecting a test runtime without loading the real artifact.
- `NFR-API-002` Controllers shall depend on application abstractions rather than transport adapters.

### Security and Privacy

- `NFR-SEC-001` Uploaded images shall be treated as untrusted input.
- `NFR-SEC-002` The public API shall enforce file-size, content-type, decoding, and request-time limits.
- `NFR-SEC-003` Raw inspection images shall not be retained or logged by default.
- `NFR-SEC-004` PyTorch artifacts shall be trusted deployment inputs and shall not come from arbitrary users.
- `NFR-SEC-005` Secrets and environment-specific paths shall not be committed.
- `NFR-SEC-006` The Python service shall remain internal or use explicit protection if exposed beyond a trusted boundary.
- `NFR-SEC-007` Production error responses shall not disclose sensitive paths or internals.

### Documentation

- `NFR-MOD-007` Documentation shall distinguish verified facts, exploratory measurements, selected decisions, and plans.
- `NFR-MOD-008` Major architectural or model decisions shall be reflected before a release milestone.

## Acceptance Criteria

### Python Model MVP – Achieved

- `AC-MOD-001` A normal-only feature memory builds from a deterministic manifest.
- `AC-MOD-002` Normal and anomalous images are scored without gradient tracking.
- `AC-INF-001` Patch scores and an image-level decision are produced.
- `AC-EVA-001` Image metrics and defect-group results are reported.
- `AC-VIS-001` Overlays can be generated for normal and anomalous images.
- `AC-MOD-003` Automated tests, compilation, and dependency checks pass.

### Artifact and Local Inference MVP – Achieved

- `AC-ART-001` A Capsule 320 x 320 artifact can be exported and loaded.
- `AC-ART-002` Loaded memory and metadata match export configuration.
- `AC-INF-002` The CLI classifies a known normal Capsule image as normal.
- `AC-INF-003` The CLI classifies a known anomalous Capsule image as anomalous.
- `AC-INF-004` Prediction returns a 40 x 40 patch-score grid.
- `AC-INF-005` Path and stream inference produce the same verified score, threshold, and decision.

### Internal Inference Service MVP – Achieved

- `AC-SVC-001` The service starts with a configured real artifact.
- `AC-SVC-002` The artifact and extractor initialize during startup.
- `AC-SVC-003` Liveness returns a healthy response.
- `AC-SVC-004` A multipart anomalous Capsule image returns the expected model identity and anomalous decision.
- `AC-SVC-005` Service construction supports an injected runtime for automated tests.
- `AC-SVC-010` A successful prediction includes a decodable threshold-normalized RGB PNG heatmap with the configured model input dimensions.

### Backend Integration MVP – Achieved

- `AC-API-001` A versioned backend endpoint accepts a valid image and returns the defined result.
- `AC-API-002` Missing, unsupported, and oversized uploads are rejected according to backend policy.
- `AC-API-003` The backend communicates through an application abstraction and HTTP adapter.
- `AC-API-004` A real Capsule image completes the C#-to-Python-to-model request successfully.
- `AC-API-005` The backend result matches the authoritative Python score, threshold, and decision for the verified image.

### Service Hardening Milestone – Pending

- `AC-SVC-006` Readiness reflects loaded runtime availability.
- `AC-SVC-007` Malformed images return a stable structured internal error.
- `AC-SVC-008` Timeout, cancellation, and concurrency behavior are documented and tested.
- `AC-SVC-009` Both processes can be started reproducibly from documented configuration.

### Model Evaluation Expansion – Pending

- `AC-EVA-002` Another category is evaluated through the generic workflow without category-specific model code.
- `AC-EVA-003` Pixel localization metrics are implemented against benchmark masks.
- `AC-EVA-004` Final procedures avoid configuration selection on the reported test partition or use a new untouched holdout.

### Optional Runtime-Portability Milestone – Pending

- `AC-ART-003` A selected feature extractor is exported to a portable format when a concrete runtime need exists.
- `AC-ART-004` Numerical parity is verified for that supported portable path.
- `AC-ART-005` Supporting preprocessing and model state can be reconstructed in the target runtime.

### Desktop Client MVP – Achieved

- `AC-DSK-001` The desktop client submits an image and displays the backend result.
- `AC-DSK-002` The desktop client reports backend liveness and inference readiness.
- `AC-DSK-003` Normal and anomalous Capsule images complete the verified desktop-to-backend analysis workflow.

### Web Client MVP – Pending

- `AC-WEB-001` The web client submits an image and displays the backend result.
- `AC-API-006` Web and desktop clients use the same versioned backend contract.

## Known Reference Results

Results are exploratory because MVTec AD test data was inspected during development.

### Bottle, 224 x 224, Complete Memory

- maximum: accuracy `0.9398`, precision `1.0000`, recall `0.9206`, F1 `0.9587`, 0 FP, 5 FN;
- top 1% mean: accuracy, precision, recall, and F1 `1.0000`, 0 FP, 0 FN;
- top 5% mean: accuracy `0.9759`, precision `0.9692`, recall `1.0000`, F1 `0.9844`, 2 FP, 0 FN.

### Capsule, 320 x 320, Complete Memory, Top 1% Mean

- true positives: 104;
- true negatives: 21;
- false positives: 2;
- false negatives: 5;
- accuracy: `0.9470`;
- precision: `0.9811`;
- recall: `0.9541`;
- F1: `0.9674`.

Random sampling reduced memory and runtime but also reduced Capsule recall. Complete memory remains selected.

## Known Limitations and Risks

- Current artifacts depend on trusted PyTorch tensor serialization.
- Artifact metadata does not fully describe all preprocessing semantics and pretrained-weight identity.
- Exact nearest-neighbor search over complete memory is computationally and memory intensive.
- Multiple service workers would duplicate approximately 410.16 MiB of feature memory each.
- The current process-local prediction lock limits concurrency.
- The threshold rule may not generalize equally across categories or deployment conditions.
- Test data influenced exploratory decisions, so benchmark numbers are not unbiased final estimates.
- Heatmaps have not been validated through pixel-level metrics.
- A model fitted on one category shall not be assumed to work on another.
- Lighting, alignment, camera, material, and production changes may cause errors.
- HTTP and process boundaries add deployment and lifecycle complexity.
- The internal service lacks production hardening and readiness behavior.
- MVTec licenses restrict commercial use and redistribution.
- The web client does not yet exist.
- The backend and desktop client do not yet transport or display the service-generated heatmap.
- The system is not validated for autonomous production decisions.

## Open Decisions

- Which additional MVTec AD categories should be evaluated next?
- Which pixel-level metrics should become localization acceptance measures?
- Should optimization use a principled coreset, indexed search, or both?
- How should complete preprocessing and weight identity be represented in artifact metadata?
- Which checksum and compatibility controls are required?
- What readiness, timeout, cancellation, and retry policies should apply?
- How should the Python service be packaged, supervised, and network-isolated?
- How should heatmaps or localization references enter the public backend response?
- When should web-client implementation begin?

## Delivery Stages

1. **Dataset and model foundation – completed:** validation, splits, preprocessing, frozen extraction, embeddings, memory, scoring, and tests.
2. **Evaluation and visualization baseline – completed:** metrics, aggregation comparison, grouped analysis, and overlays.
3. **Python artifact and inference MVP – completed:** artifact, exporter, loader, inference APIs, and CLI.
4. **Internal inference-service MVP – completed:** FastAPI runtime, liveness, prediction contract, and tests.
5. **Backend integration MVP – completed:** ASP.NET Core analysis endpoint, upload validation, Python adapter, and verified end-to-end request.
6. **WPF desktop-client MVP – completed:** health status, local image selection, preview, backend submission, and result display.
7. **Heatmap integration – active:** propagate localization data through the backend and display it in the desktop client.
8. **Service and backend hardening – planned:** readiness, structured errors, timeouts, logging, packaging, and contract coverage.
9. **Web-client MVP – planned:** separate web client using the same backend contract.
10. **Portfolio release expansion – planned:** demonstrations, Model Card, updated releases, and clearly scoped limitations.

## Related Documentation

- `README.md` – repository introduction and setup
- `DevelopmentStatus.md` – verified progress and current next steps
- `ArchitectureOverview.md` – components, responsibilities, and data flow
- `ModelDevelopmentStrategy.md` – experimentation and model-selection strategy
- `DatasetDocumentation.md` – dataset sources, licenses, inventories, and validation
- `COMMITS.md` – commit-message conventions

## Last Updated

2026-08-17
