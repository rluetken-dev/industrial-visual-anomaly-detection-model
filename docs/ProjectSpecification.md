# Industrial Visual Anomaly Detection – Project Specification

## Document Purpose

This document defines the current scope, requirements, business rules, quality expectations, acceptance criteria, limitations, and delivery stages for the Industrial Visual Anomaly Detection system.

The system spans:

- Python model fitting, evaluation, artifacts, and inference;
- an internal FastAPI inference service;
- a public ASP.NET Core backend;
- a native WPF desktop client;
- a Docker Compose orchestration repository;
- a possible future web client.

## Requirement Identification

- `FR` – Functional Requirement
- `BR` – Business Rule
- `NFR` – Non-Functional Requirement
- `AC` – Acceptance Criterion

Requirement groups:

- `DAT` – source data and discovery
- `SPL` – fitting and validation partitions
- `PRE` – preprocessing
- `MOD` – model fitting and scoring
- `EVA` – evaluation
- `VIS` – heatmaps and visualization
- `ART` – artifacts
- `SVC` – internal Python service
- `API` – ASP.NET Core backend
- `DSK` – desktop client
- `STK` – Docker stack
- `OBS` – observability
- `SEC` – security and safety

## Project Overview

The system performs unsupervised visual anomaly detection from normal product images:

1. normal PNG or JPEG images are discovered from an external directory or resolved through a controlled manifest;
2. images are assigned reproducibly to fitting and normal-validation partitions;
3. images are converted to RGB, resized, converted to tensors, and ImageNet-normalized;
4. a frozen pretrained ResNet18 extracts `layer2` and `layer3` feature maps;
5. fused feature maps become 384-dimensional local patch embeddings;
6. embeddings from normal fitting images form a category-specific feature memory;
7. validation images are scored against that memory;
8. the maximum normal-validation image score becomes the threshold;
9. new images receive patch scores, an image score, a normal/anomalous decision, and an anomaly heatmap.

The system detects deviations from learned normal appearance. It does not identify the precise defect type.

Python remains authoritative for model fitting and inference. FastAPI exposes the internal inference boundary. ASP.NET Core owns the public application contract. The WPF client consumes only the backend. Docker Compose runs the backend and inference service with a selected local artifact.

## Current Implementation Status

### Implemented and Verified

- validation tooling for MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- external recursive PNG/JPEG discovery;
- deterministic automatic fitting and validation splits;
- portable `training_split.json` persistence;
- versioned MVTec manifest compatibility;
- deterministic preprocessing;
- frozen ResNet18 feature extraction;
- patch embeddings and feature-memory fitting;
- exact chunked nearest-neighbor scoring;
- top-fraction image aggregation;
- normal-validation threshold selection;
- exploratory Bottle and Capsule evaluation;
- heatmap generation and Base64 PNG encoding;
- Python/PyTorch artifact export, validation, loading, and inference;
- persistent FastAPI runtime;
- ASP.NET Core health, validation, analysis, and error contracts;
- WPF health display, image preview, analysis results, and interactive heatmap overlay;
- Docker Compose image builds, networking, health checks, artifact mounting, and local verification;
- 92 passing Python tests plus separate backend, desktop, and stack CI.

### Reference Evidence

- Capsule 320 x 320 artifact: 280,000 x 384 feature memory, approximately 410.16 MiB, threshold `2.501821517944336`;
- generalized Bottle 320 x 320 artifact: 267,200 x 384 feature memory, approximately 391.41 MiB, threshold `3.2163138389587402`;
- the refactored Capsule exporter reproduced the established feature memory byte-for-byte;
- a known normal Bottle image was classified as normal;
- a known defective Bottle image was classified as anomalous;
- the complete client-to-backend-to-Python heatmap workflow is verified;
- the backend and inference service start from a clean stack clone.

### Not Yet Implemented

- general evaluation of an artifact against arbitrary normal and anomalous test directories;
- multi-artifact loading or explicit category selection;
- quantitative pixel-level localization metrics;
- complete preprocessing and pretrained-weight provenance in artifact metadata;
- artifact checksums and signatures;
- a non-square preprocessing policy;
- a web client;
- authentication and production deployment hardening;
- public artifact distribution.

## Project Goals

- accept normal product images without requiring dataset-specific code;
- fit category-specific normal references without defect examples;
- preserve strict fitting, validation, and test separation;
- make partitions and artifacts reproducible;
- expose persistent CPU-compatible inference;
- provide interpretable anomaly heatmaps;
- keep public APIs independent from Python and UI implementation details;
- support native desktop analysis and future clients;
- enable reproducible local backend and inference startup.

## Intended Users

- developers fitting anomaly models from normal image collections;
- researchers comparing preprocessing, memory, scoring, and thresholds;
- backend and client developers consuming stable contracts;
- portfolio reviewers evaluating architecture and reproducibility;
- desktop users submitting inspection images and reviewing results.

## Scope

### Python Model Scope

Image discovery, deterministic splitting, preprocessing, feature extraction, feature-memory fitting, anomaly scoring, threshold derivation, evaluation, visualization, artifact export and loading, path and stream inference, and the internal FastAPI service.

### Backend Scope

Public image-analysis API, upload validation, Problem Details, trace identifiers, health and readiness, Python-service communication, cancellation, timeout behavior, and client-neutral response mapping.

### Desktop Scope

Backend status, local image selection, preview, analysis submission, result presentation, and interactive heatmap visibility and opacity.

### Stack Scope

Version-pinned backend and inference builds, Docker Compose networking, startup ordering, health checks, read-only artifact mounting, host ports, and verification scripts.

### Explicitly Deferred

- supervised defect classification;
- online or continual learning;
- automatic retraining;
- database persistence;
- untrusted artifact uploads;
- certified production inspection;
- commercial dataset use without separate legal review.

## Functional Requirements

### Source Data and Discovery

- `FR-DAT-001` The model shall accept source directories outside the repository.
- `FR-DAT-002` The generalized fitting workflow shall recursively discover PNG, JPG, and JPEG images case-insensitively.
- `FR-DAT-003` Missing source directories shall be rejected.
- `FR-DAT-004` Source directories without supported images shall be rejected.
- `FR-DAT-005` Unsupported files shall not become fitting inputs.
- `FR-DAT-006` Dataset archives, extracted images, and generated outputs shall remain excluded from Git.
- `FR-DAT-007` MVTec validation tools shall remain available for controlled benchmark datasets.
- `FR-DAT-008` Dataset identity and category shall be explicit artifact inputs.

### Fitting and Validation Splits

- `FR-SPL-001` Automatic splitting shall require at least two unique image paths.
- `FR-SPL-002` Validation fraction shall be greater than zero and less than one.
- `FR-SPL-003` Splitting shall use an explicit deterministic seed.
- `FR-SPL-004` Fitting and validation partitions shall both remain non-empty.
- `FR-SPL-005` Partitions shall not overlap and shall cover every discovered source image.
- `FR-SPL-006` The generalized exporter shall write `training_split.json` after successful artifact creation.
- `FR-SPL-007` Split records shall use paths relative to the supplied source directory.
- `FR-SPL-008` Existing versioned manifest workflows shall remain supported.
- `FR-SPL-009` Manifest validation shall reject duplicate, overlapping, absolute, parent-traversing, or unresolved paths.

### Image Preprocessing

- `FR-PRE-001` Images shall be converted to RGB.
- `FR-PRE-002` Images shall be resized directly to the configured square input size.
- `FR-PRE-003` Images shall be converted to tensors and ImageNet-normalized.
- `FR-PRE-004` Fitting, validation, evaluation, CLI, and service inference shall share preprocessing semantics.
- `FR-PRE-005` Input size shall be positive and divisible by 32.
- `FR-PRE-006` Input size shall be stored in artifact metadata.

### Model Fitting

- `FR-MOD-001` A pretrained ResNet18 shall run frozen in evaluation mode.
- `FR-MOD-002` `layer2` and `layer3` features shall be fused into 384-dimensional embeddings.
- `FR-MOD-003` Feature memory shall contain only normal fitting embeddings.
- `FR-MOD-004` Feature-memory construction shall preserve deterministic loader order.
- `FR-MOD-005` Memory sampling, when enabled, shall use an explicit fraction and seed.
- `FR-MOD-006` One artifact shall represent one declared category.
- `FR-MOD-007` Shared training orchestration shall be independent from CLI and dataset-specific directory conventions.
- `FR-MOD-008` Invalid configuration values shall fail before expensive fitting begins.

### Anomaly Inference

- `FR-INF-001` Query patches shall be compared with the fitted memory using exact Euclidean nearest-neighbor distance.
- `FR-INF-002` Memory search shall support configurable chunking.
- `FR-INF-003` Patch scores shall retain the artifact-defined spatial grid.
- `FR-INF-004` Image scoring shall support top-fraction mean aggregation.
- `FR-INF-005` Scores strictly above the stored threshold shall be anomalous.
- `FR-INF-006` Path and binary-stream inference shall use the same model behavior.
- `FR-INF-007` Inference shall derive category, input size, grid, aggregation, and threshold from the artifact.

### Threshold and Evaluation

- `FR-EVA-001` Thresholds shall derive only from held-out normal validation images.
- `FR-EVA-002` The current threshold rule shall use the maximum normal-validation image score.
- `FR-EVA-003` Test images and labels shall not influence fitting or threshold calculation.
- `FR-EVA-004` Labeled evaluation shall report confusion counts, accuracy, precision, recall, and F1.
- `FR-EVA-005` Defect-group evaluation shall report group score distributions and detection rates.
- `FR-EVA-006` Results influenced by inspected test data shall be labeled exploratory.
- `FR-EVA-007` A future general evaluator shall score an existing artifact against optional normal and anomalous test directories without refitting.
- `FR-EVA-008` Future pixel evaluation shall use explicitly selected mask-based metrics.

### Visualization

- `FR-VIS-001` Patch-score grids shall support resizing to model-input dimensions.
- `FR-VIS-002` Heatmaps shall support threshold-based normalization.
- `FR-VIS-003` The service shall produce RGB PNG heatmaps.
- `FR-VIS-004` Internal HTTP transport shall encode heatmaps as Base64 text.
- `FR-VIS-005` Heatmap contracts shall include content type, width, height, and encoded data.
- `FR-VIS-006` Heatmap generation shall not alter classification results.

### Model Artifacts

- `FR-ART-001` A core artifact shall contain `metadata.json` and `feature_memory.pt`.
- `FR-ART-002` Generalized directory export shall additionally contain `training_split.json`.
- `FR-ART-003` Metadata shall include schema, dataset, category, backbone, input and grid sizes, embedding dimension, aggregation, threshold, sampling configuration, and memory entry count.
- `FR-ART-004` Artifact writing shall validate tensor dimensions, finiteness, entry count, and embedding dimension.
- `FR-ART-005` Artifact loading shall use CPU loading and `weights_only=True`.
- `FR-ART-006` Generated artifacts shall remain excluded from Git.
- `FR-ART-007` Current artifacts shall be identified as trusted Python/PyTorch deployment inputs.
- `FR-ART-008` Existing manifest-based exports shall remain behaviorally compatible after shared-training refactoring.

### Internal Python Service

- `FR-SVC-001` The service shall read the selected artifact path from controlled configuration.
- `FR-SVC-002` The artifact and extractor shall load once during startup.
- `FR-SVC-003` Requests shall reuse the loaded runtime.
- `FR-SVC-004` `GET /health/live` shall report process liveness.
- `FR-SVC-005` `POST /api/v1/predictions` shall accept multipart field `image`.
- `FR-SVC-006` Missing and unreadable images shall be rejected.
- `FR-SVC-007` Successful responses shall contain model ID, category, score, threshold, Boolean decision, and heatmap.
- `FR-SVC-008` Prediction execution shall protect shared runtime state with an explicit concurrency policy.
- `FR-SVC-009` The current runtime shall load one artifact; multi-artifact selection is a future milestone.

### ASP.NET Core Backend

- `FR-API-001` The backend shall expose `POST /api/v1/analyses`.
- `FR-API-002` The backend shall validate image size, supported content type, and file signature.
- `FR-API-003` The backend shall expose liveness and inference-aware readiness.
- `FR-API-004` Public errors shall use Problem Details with trace identifiers.
- `FR-API-005` The backend shall call the internal Python service through an application abstraction.
- `FR-API-006` The backend shall map model identity, score, threshold, decision, duration, trace ID, and heatmap to its public contract.
- `FR-API-007` Clients shall consume the backend rather than the internal Python service.

### WPF Desktop Client

- `FR-DSK-001` The client shall display backend liveness and inference readiness.
- `FR-DSK-002` The client shall allow local PNG or JPEG selection and preview.
- `FR-DSK-003` The client shall submit the selected image to the backend.
- `FR-DSK-004` The client shall display decision, score, threshold, model, category, duration, and trace ID.
- `FR-DSK-005` The client shall decode and overlay the returned heatmap.
- `FR-DSK-006` The client shall allow heatmap visibility and opacity control.
- `FR-DSK-007` UI-specific colors and interaction shall remain desktop concerns.

### Docker Stack

- `FR-STK-001` The stack shall build backend and inference images from versioned source refs.
- `FR-STK-002` Compose shall connect services through an internal network.
- `FR-STK-003` Backend startup shall depend on healthy inference.
- `FR-STK-004` The selected artifact shall be mounted read-only.
- `FR-STK-005` Host ports shall be configurable.
- `FR-STK-006` A verification script shall check liveness, readiness, analysis, and heatmap decoding.
- `FR-STK-007` The WPF client shall remain outside Linux containers.

## Business Rules

- `BR-MOD-001` Only normal fitting images may populate feature memory.
- `BR-MOD-002` Unrelated categories shall use separate artifacts.
- `BR-EVA-001` Normal validation images may determine thresholds but may not populate feature memory.
- `BR-EVA-002` Test labels may support evaluation but not fitting or threshold selection.
- `BR-ART-001` Released artifacts shall be immutable.
- `BR-ART-002` Artifact identity shall remain traceable to category, configuration, and source revision.
- `BR-CLI-001` User-facing scripts shall not duplicate core fitting logic.
- `BR-API-001` The backend decision shall remain authoritative for clients.
- `BR-SAF-001` Heatmaps shall not be represented as certified defect segmentation.

## Non-Functional Requirements

### Reproducibility

- `NFR-REP-001` Discovery order shall be deterministic.
- `NFR-REP-002` Split and sampling seeds shall be explicit.
- `NFR-REP-003` Exact generalized split membership shall be persisted.
- `NFR-REP-004` Dependencies and supported Python version shall be recorded.
- `NFR-REP-005` Existing reference exports shall be regression-checked during refactoring.

### Performance and Resource Use

- `NFR-PER-001` CPU-only execution shall remain supported.
- `NFR-PER-002` Nearest-neighbor calculation shall support chunking.
- `NFR-PER-003` The service shall not reload the artifact per request.
- `NFR-PER-004` Memory-reduction strategies shall be evaluated against complete memory before selection.

### Maintainability and Testability

- `NFR-MNT-001` Dataset, model, artifact, inference, service, backend, and UI concerns shall remain separated.
- `NFR-MNT-002` Constructor and public input invariants shall fail early.
- `NFR-MNT-003` Core deterministic components shall have automated tests.
- `NFR-MNT-004` Service construction shall support injected test runtimes.
- `NFR-MNT-005` Documentation shall distinguish verified behavior from future work.

### Security and Safety

- `NFR-SEC-001` Dataset and artifact paths shall come from controlled configuration.
- `NFR-SEC-002` User-controlled paths shall not escape intended roots.
- `NFR-SEC-003` PyTorch artifacts shall be trusted inputs, not arbitrary uploads.
- `NFR-SEC-004` Public upload validation shall occur at the backend boundary.
- `NFR-SEC-005` Raw image content shall not be logged.
- `NFR-SEC-006` The system shall not claim production certification.

## Acceptance Criteria

### Generalized Fitting Milestone – Achieved

- `AC-GEN-001` PNG and JPEG images are discovered recursively from an external directory.
- `AC-GEN-002` A seeded automatic split produces non-overlapping fitting and validation partitions.
- `AC-GEN-003` Exact membership is stored with relative paths in `training_split.json`.
- `AC-GEN-004` A Bottle artifact is exported without an MVTec manifest.
- `AC-GEN-005` A known normal Bottle image is classified as normal.
- `AC-GEN-006` A known defective Bottle image is classified as anomalous.
- `AC-GEN-007` The established Capsule feature memory remains byte-for-byte reproducible after refactoring.
- `AC-GEN-008` All 92 Python tests pass.

### Artifact and Inference MVP – Achieved

- `AC-ART-001` Category artifacts can be exported, loaded, and used without fitting data.
- `AC-ART-002` Path and stream inference share numerical behavior.
- `AC-ART-003` Invalid artifact contents are rejected.
- `AC-ART-004` Inference returns score, threshold, decision, and patch scores.

### Service and Backend MVP – Achieved

- `AC-SVC-001` FastAPI loads the selected artifact and extractor during startup.
- `AC-SVC-002` Multipart predictions return model identity, decision data, and a decodable PNG heatmap.
- `AC-API-001` The backend verifies Python readiness and maps successful analysis responses.
- `AC-API-002` Invalid uploads and inference failures use stable public errors.

### Desktop Heatmap MVP – Achieved

- `AC-DSK-001` The desktop client selects and previews an image.
- `AC-DSK-002` Normal and anomalous images complete the backend-driven workflow.
- `AC-DSK-003` The client displays analysis details and trace identity.
- `AC-DSK-004` The client overlays the heatmap and supports visibility and opacity controls.

### Docker Stack MVP – Achieved

- `AC-STK-001` Version-pinned backend and inference images build successfully.
- `AC-STK-002` Compose reaches healthy inference and backend readiness.
- `AC-STK-003` A clean clone completes real Capsule analysis and heatmap verification with a local artifact.
- `AC-STK-004` Teardown and restart reproduce the healthy state.

### General Evaluation Expansion – Pending

- `AC-EVA-001` An exported artifact can be evaluated against arbitrary normal and anomalous test directories without refitting.
- `AC-EVA-002` The generalized workflow is evaluated on a non-MVTec image collection.
- `AC-EVA-003` Minimum recommended normal-image counts are documented from evidence.

### Multi-Artifact Milestone – Pending

- `AC-MUL-001` Multiple category artifacts have an explicit selection mechanism.
- `AC-MUL-002` Service, backend, and client contracts define category routing without ambiguity.
- `AC-MUL-003` Readiness reports the selected or available artifact state.

### Public Artifact Release – Pending

- `AC-REL-001` Dataset redistribution rights are confirmed.
- `AC-REL-002` Artifact checksums and provenance are published.
- `AC-REL-003` Results link to the exact artifact.
- `AC-REL-004` A Model Card documents intended use, limitations, and evaluation.

## Known Limitations and Risks

- artifacts depend on trusted PyTorch tensor serialization;
- feature memories are large and exact search is compute-intensive;
- multiple service workers duplicate feature memory;
- maximum-normal threshold selection may be unstable for small validation sets;
- direct square resizing may distort non-square products;
- current artifact metadata does not fully describe preprocessing or pretrained weights;
- the service currently loads one artifact;
- generalized fitting has not yet been evaluated on a truly external non-MVTec dataset;
- Bottle smoke tests do not replace complete evaluation;
- heatmaps lack quantitative pixel-level validation;
- Base64 heatmaps increase response size;
- authentication and production hardening are incomplete;
- MVTec licenses constrain redistribution and commercial use;
- the system is not certified for production decisions.

## Open Decisions

- What external folder conventions should be recommended beyond a normal-image directory?
- What minimum normal-image count is useful for different category complexity?
- How should small datasets calibrate thresholds safely?
- How should non-square products be resized or padded?
- How should multiple artifacts be selected and versioned at runtime?
- Which artifact checksums and provenance fields become mandatory?
- Which coverage-preserving memory-reduction method should be evaluated first?
- Which pixel-level localization metrics should be adopted?

## Delivery Stages

1. **Dataset and model foundation – completed.**
2. **Bottle and Capsule exploratory evaluation – completed.**
3. **Python artifact and inference MVP – completed.**
4. **FastAPI and ASP.NET Core integration – completed.**
5. **WPF analysis and heatmap MVP – completed.**
6. **Docker Compose local stack – completed.**
7. **General external-directory fitting – completed.**
8. **General artifact evaluation – next.**
9. **Multi-artifact category selection – planned.**
10. **Production and public artifact hardening – future.**

## Related Documentation

- `ArchitectureOverview.md` – system boundaries and runtime flow
- `DatasetDocumentation.md` – dataset sources, licenses, inventories, and validation
- `DevelopmentStatus.md` – verified implementation status
- `ModelDevelopmentStrategy.md` – fitting, validation, evaluation, and approval rules
- a future `ModelCard.md` – released artifact behavior and limitations

## Last Updated

2026-08-19
