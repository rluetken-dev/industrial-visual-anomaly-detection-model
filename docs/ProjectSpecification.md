# Industrial Visual Anomaly Detection – Project Specification

## Document Purpose

This document defines the stable scope, requirements, business rules, quality expectations, acceptance criteria, limitations, and delivery stages for the Industrial Visual Anomaly Detection system.

Implementation progress belongs in `DevelopmentStatus.md`. Architecture and runtime flow belong in `ArchitectureOverview.md`.

## System Scope

The system spans:

- Python model fitting, evaluation, artifacts, registry handling, and inference;
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
- `REG` – model registry and runtime selection
- `SVC` – internal Python service
- `API` – ASP.NET Core backend
- `DSK` – desktop client
- `STK` – Docker stack
- `SEC` – security and safety

## Project Overview

The system performs unsupervised visual anomaly detection from normal product images:

1. normal PNG or JPEG images are discovered or resolved through a controlled manifest;
2. images are assigned reproducibly to fitting and validation partitions;
3. images are converted to RGB, resized, converted to tensors, and normalized;
4. a frozen pretrained ResNet18 extracts `layer2` and `layer3` features;
5. fused features become 384-dimensional local patch embeddings;
6. fitting embeddings form a category-specific feature memory;
7. validation images are scored against that memory;
8. a configured normal-score quantile becomes the threshold;
9. new images receive patch scores, an image score, a decision, and a heatmap.

Each exported artifact represents one model category and configuration. A registry can compose multiple independent artifacts into one deployed inference-service instance. Callers discover the enabled catalog and select a model through a stable identifier.

Python remains authoritative for fitting, registry validation, and inference. FastAPI exposes the internal boundary. ASP.NET Core owns the public contract. Clients consume only the backend. Docker Compose runs backend and inference with a read-only external registry and artifacts.

The system detects deviations from learned normal appearance. It does not identify the exact defect type or automatically recognize which product category appears in an image.

## Project Goals

- accept normal product images without dataset-specific core logic;
- fit category-specific normal references without defect examples;
- preserve strict fitting, validation, and test separation;
- make partitions, artifacts, and deployment model sets reproducible;
- expose persistent CPU-compatible inference;
- load and select multiple category artifacts safely;
- provide interpretable anomaly heatmaps;
- keep public APIs independent from Python and UI details;
- support native desktop analysis and future clients;
- enable reproducible local backend and inference startup.

## Scope Boundaries

### Python Model Scope

Image discovery, splitting, preprocessing, feature extraction, memory fitting, scoring, thresholds, evaluation, visualization, artifact export and loading, model-registry validation, runtime selection, path and stream inference, and FastAPI.

### Backend Scope

Public model catalog and analysis APIs, upload and model-ID validation, Problem Details, traces, health and readiness, Python communication, cancellation, timeouts, and client-neutral mapping.

### Desktop Scope

Backend status, catalog retrieval, model selection, local image selection, preview, analysis submission, result presentation, and heatmap interaction.

### Stack Scope

Explicit backend and inference source builds, Docker Compose networking, ordering, health checks, read-only registry and artifact mounting, host ports, and verification scripts.

### Explicitly Deferred

- supervised defect classification;
- automatic visual category recognition;
- online or continual learning;
- automatic retraining;
- dynamic registry hot reload;
- lazy loading and unloading of models;
- database persistence;
- untrusted artifact uploads;
- certified production inspection;
- commercial dataset use without legal review.

## Functional Requirements

### Source Data and Discovery

- `FR-DAT-001` The model shall accept source directories outside the repository.
- `FR-DAT-002` General fitting shall discover PNG, JPG, and JPEG images recursively and case-insensitively.
- `FR-DAT-003` Missing or empty source directories shall be rejected.
- `FR-DAT-004` Unsupported files shall not become fitting inputs.
- `FR-DAT-005` Datasets and generated outputs shall remain excluded from Git.
- `FR-DAT-006` MVTec validation tools shall remain available.
- `FR-DAT-007` Dataset identity and category shall be explicit artifact inputs.

### Fitting and Validation Splits

- `FR-SPL-001` Automatic splitting shall require at least two unique paths.
- `FR-SPL-002` Validation fraction shall be strictly between zero and one.
- `FR-SPL-003` Splitting shall use an explicit deterministic seed.
- `FR-SPL-004` Both partitions shall remain non-empty.
- `FR-SPL-005` Partitions shall not overlap and shall cover all source images.
- `FR-SPL-006` General export shall write `training_split.json` after successful artifact creation.
- `FR-SPL-007` Split records shall use source-relative paths.
- `FR-SPL-008` Versioned manifest workflows shall remain supported.
- `FR-SPL-009` Manifests shall reject duplicates, overlap, absolute paths, traversal, and unresolved paths.

### Image Preprocessing

- `FR-PRE-001` Images shall be converted to RGB.
- `FR-PRE-002` Images shall be resized directly to the configured square input size.
- `FR-PRE-003` Images shall be converted to tensors and ImageNet-normalized.
- `FR-PRE-004` Fitting, validation, evaluation, CLI, and service inference shall share semantics.
- `FR-PRE-005` Input size shall be positive and divisible by 32.
- `FR-PRE-006` Input size shall be stored in artifact metadata.

### Model Fitting and Inference

- `FR-MOD-001` A pretrained ResNet18 shall run frozen in evaluation mode.
- `FR-MOD-002` `layer2` and `layer3` shall form 384-dimensional embeddings.
- `FR-MOD-003` Feature memory shall contain only normal fitting embeddings.
- `FR-MOD-004` Feature-memory construction shall preserve deterministic order.
- `FR-MOD-005` Sampling shall use an explicit fraction and seed.
- `FR-MOD-006` One artifact shall represent one declared category.
- `FR-MOD-007` Training orchestration shall remain independent from CLI and dataset conventions.
- `FR-MOD-008` Invalid configuration shall fail before expensive fitting.
- `FR-MOD-009` Query patches shall use exact Euclidean nearest-neighbor distance.
- `FR-MOD-010` Search shall support configurable chunking.
- `FR-MOD-011` Patch scores shall retain the artifact-defined grid.
- `FR-MOD-012` Image scoring shall support top-fraction mean aggregation.
- `FR-MOD-013` Scores strictly above the stored threshold shall be anomalous.
- `FR-MOD-014` Path and stream inference shall share model behavior.
- `FR-MOD-015` Inference shall derive category, sizes, aggregation, and threshold from the artifact.

### Threshold and Evaluation

- `FR-EVA-001` Initial thresholds shall derive only from held-out normal validation images.
- `FR-EVA-002` Threshold selection shall use a configurable normal-score quantile.
- `FR-EVA-003` Quantile `1.0` shall preserve maximum-normal behavior.
- `FR-EVA-004` Test images and labels shall not influence fitting or initial thresholds.
- `FR-EVA-005` Evaluation shall accept CSV columns `image`, `group`, and `is_anomalous`.
- `FR-EVA-006` Manifest paths shall resolve relative to an explicit dataset root.
- `FR-EVA-007` Invalid labels, duplicates, missing files, unsupported suffixes, absolute paths, and traversal shall be rejected.
- `FR-EVA-008` Evaluation shall score an existing artifact without refitting.
- `FR-EVA-009` Evaluation shall report distributions, confusion counts, accuracy, precision, recall, specificity, F1, group rates, and error lists.
- `FR-EVA-010` Calibration results shall not be claimed as independent final evaluation.

### Visualization

- `FR-VIS-001` Patch grids shall support resizing to model-input dimensions.
- `FR-VIS-002` Heatmaps shall support threshold-based normalization.
- `FR-VIS-003` The service shall produce RGB PNG heatmaps.
- `FR-VIS-004` HTTP transport shall encode heatmaps as Base64.
- `FR-VIS-005` Contracts shall include content type, width, height, and data.
- `FR-VIS-006` Heatmap generation shall not alter classification.

### Model Artifacts

- `FR-ART-001` A core artifact shall contain `metadata.json` and `feature_memory.pt`.
- `FR-ART-002` General export shall additionally contain `training_split.json`.
- `FR-ART-003` Schema-version-2 metadata shall record model, threshold, and sampling configuration.
- `FR-ART-004` Writing shall validate dimensions, finiteness, entry count, and embedding dimension.
- `FR-ART-005` Loading shall use CPU and `weights_only=True`.
- `FR-ART-006` Generated artifacts shall remain excluded from Git.
- `FR-ART-007` Artifacts shall be treated as trusted Python/PyTorch deployment inputs.
- `FR-ART-008` Manifest exports shall remain behaviorally compatible after refactoring.
- `FR-ART-009` Schema-version-1 artifacts shall load through maximum-normal defaults.

### Model Registry and Runtime Selection

- `FR-REG-001` Registry mode shall read a JSON document from controlled configuration.
- `FR-REG-002` The registry shall declare a supported schema version.
- `FR-REG-003` The registry shall declare one default model identifier.
- `FR-REG-004` Each entry shall declare ID, display name, relative artifact directory, and enabled state.
- `FR-REG-005` Model identifiers and artifact directories shall be unique.
- `FR-REG-006` Model identifiers shall follow a restricted stable format.
- `FR-REG-007` Artifact directories shall be relative and shall not traverse above the registry directory.
- `FR-REG-008` At least one model shall be enabled.
- `FR-REG-009` The default model shall reference an enabled entry.
- `FR-REG-010` Every enabled artifact directory shall exist.
- `FR-REG-011` Disabled missing artifacts may be accepted because they are not loaded.
- `FR-REG-012` Unexpected fields and invalid structure shall be rejected.
- `FR-REG-013` Every enabled artifact shall load into a separate persistent runtime during startup.
- `FR-REG-014` Loaded runtime identifiers shall match enabled registry identifiers.
- `FR-REG-015` Available models shall preserve configured order.
- `FR-REG-016` Catalog category and input size shall derive from loaded artifact metadata.
- `FR-REG-017` Omitted model selection shall resolve to the configured default.
- `FR-REG-018` Explicit valid selection shall resolve to the matching runtime.
- `FR-REG-019` Empty and unknown model identifiers shall be rejected clearly.

### Internal Python Service

- `FR-SVC-001` Exactly one of registry mode and legacy artifact mode shall be configured.
- `FR-SVC-002` Legacy mode shall preserve one-artifact compatibility.
- `FR-SVC-003` Configured artifacts and extractors shall load once during startup.
- `FR-SVC-004` Requests shall reuse loaded runtimes.
- `FR-SVC-005` `GET /health/live` shall report process liveness.
- `FR-SVC-006` `GET /api/v1/models` shall expose the default and available models.
- `FR-SVC-007` Legacy mode shall expose its runtime as a one-model catalog.
- `FR-SVC-008` `POST /api/v1/predictions` shall accept multipart `image`.
- `FR-SVC-009` Predictions shall accept optional multipart `modelId`.
- `FR-SVC-010` Missing and unreadable images shall be rejected.
- `FR-SVC-011` Unknown model IDs shall return a clear not-found response.
- `FR-SVC-012` Successful responses shall contain actual model ID, category, score, threshold, decision, and heatmap.
- `FR-SVC-013` Each shared runtime shall have an explicit concurrency policy.

### ASP.NET Core Backend

- `FR-API-001` The backend shall expose `GET /api/v1/models`.
- `FR-API-002` The backend catalog shall map the inference default and available models.
- `FR-API-003` The backend shall expose `POST /api/v1/analyses`.
- `FR-API-004` Analysis shall accept an optional model identifier.
- `FR-API-005` The backend shall validate upload and model inputs.
- `FR-API-006` The selected model identifier shall be forwarded to Python.
- `FR-API-007` The backend shall expose liveness and inference-aware readiness.
- `FR-API-008` Public errors shall use Problem Details with trace identifiers.
- `FR-API-009` Public results shall include model identity, score, threshold, decision, duration, trace ID, and heatmap.
- `FR-API-010` Clients shall consume the backend rather than Python directly.

### WPF Desktop Client

- `FR-DSK-001` The client shall display backend and inference status.
- `FR-DSK-002` The client shall retrieve the model catalog from the backend.
- `FR-DSK-003` The client shall select the catalog default initially.
- `FR-DSK-004` The user shall be able to select another available model.
- `FR-DSK-005` The client shall select and preview local PNG or JPEG images.
- `FR-DSK-006` The selected model ID shall accompany analysis.
- `FR-DSK-007` The client shall display decision, score, threshold, model, category, duration, and trace ID.
- `FR-DSK-008` The client shall decode and overlay the heatmap.
- `FR-DSK-009` The client shall control heatmap visibility and opacity.
- `FR-DSK-010` UI colors and interaction shall remain desktop concerns.

### Docker Stack

- `FR-STK-001` The stack shall build compatible backend and inference revisions.
- `FR-STK-002` Compose shall connect services through an internal network.
- `FR-STK-003` Backend startup shall depend on healthy inference.
- `FR-STK-004` The registry and artifact root shall be mounted read-only.
- `FR-STK-005` The inference registry path shall reference the container path.
- `FR-STK-006` Host ports shall be configurable.
- `FR-STK-007` A verification script shall check health and optional model-specific analysis.
- `FR-STK-008` Verification shall confirm the response model matches the requested model.
- `FR-STK-009` The WPF client shall remain outside Linux containers.

## Business Rules

- `BR-MOD-001` Only normal fitting images may populate feature memory.
- `BR-MOD-002` Unrelated categories shall use separate artifacts.
- `BR-EVA-001` Normal validation images may determine thresholds but not populate memory.
- `BR-EVA-002` Test labels may support evaluation but not fitting or initial thresholds.
- `BR-EVA-003` Calibration evidence shall not be presented as untouched final evaluation.
- `BR-ART-001` Released artifacts shall be immutable.
- `BR-ART-002` Artifact identity shall remain traceable to category, configuration, and source revision.
- `BR-REG-001` The registry shall be authoritative for deployment model availability and default selection.
- `BR-REG-002` Backend and clients shall not maintain independent hard-coded model catalogs.
- `BR-REG-003` Model selection shall use stable IDs rather than display names.
- `BR-API-001` The backend decision shall remain authoritative for clients.
- `BR-SAF-001` Heatmaps shall not be represented as certified segmentation.

## Non-Functional Requirements

### Reproducibility

- `NFR-REP-001` Discovery order shall be deterministic.
- `NFR-REP-002` Split and sampling seeds shall be explicit.
- `NFR-REP-003` Exact generalized split membership shall be persisted.
- `NFR-REP-004` Dependencies and supported Python version shall be recorded.
- `NFR-REP-005` Reference exports shall be regression-checked during refactoring.
- `NFR-REP-006` Threshold method and quantile shall be persisted.
- `NFR-REP-007` Evaluation manifests shall preserve image, group, and label identity.
- `NFR-REP-008` Registry entries shall preserve stable model IDs and explicit artifact paths.

### Performance and Resource Use

- `NFR-PER-001` CPU-only execution shall remain supported.
- `NFR-PER-002` Nearest-neighbor calculation shall support chunking.
- `NFR-PER-003` The service shall not reload artifacts per request.
- `NFR-PER-004` Enabled models may load eagerly during startup.
- `NFR-PER-005` Memory-reduction strategies shall be compared with complete memory.
- `NFR-PER-006` Multiple worker processes shall be documented as duplicating loaded memories.

### Maintainability and Testability

- `NFR-MNT-001` Dataset, model, artifact, registry, inference, backend, and UI concerns shall remain separated.
- `NFR-MNT-002` Public invariants shall fail early.
- `NFR-MNT-003` Deterministic components shall have automated tests.
- `NFR-MNT-004` Service construction shall support injected test runtimes.
- `NFR-MNT-005` Documentation shall distinguish verified behavior from future work.
- `NFR-MNT-006` Legacy single-artifact behavior shall remain covered while supported.

### Security and Safety

- `NFR-SEC-001` Dataset, registry, and artifact paths shall come from controlled configuration.
- `NFR-SEC-002` User-controlled paths shall not escape intended roots.
- `NFR-SEC-003` PyTorch artifacts shall be trusted inputs, not arbitrary uploads.
- `NFR-SEC-004` Public upload validation shall occur at the backend.
- `NFR-SEC-005` Raw image content shall not be logged.
- `NFR-SEC-006` Registry and artifacts shall be mounted read-only in containers.
- `NFR-SEC-007` The system shall not claim production certification.

## Acceptance Criteria

### Generalized Fitting – Achieved

- `AC-GEN-001` External PNG and JPEG images are discovered recursively.
- `AC-GEN-002` A seeded split creates non-overlapping partitions.
- `AC-GEN-003` Relative membership is stored in `training_split.json`.
- `AC-GEN-004` A Bottle artifact is exported without an MVTec manifest.
- `AC-GEN-005` Known normal and defective Bottle images produce expected decisions.
- `AC-GEN-006` Capsule memory remains byte-for-byte reproducible after refactoring.

### Artifact, Service, Backend, and Desktop MVP – Achieved

- `AC-MVP-001` Artifacts can be exported, loaded, and used without fitting data.
- `AC-MVP-002` Path and stream inference share behavior.
- `AC-MVP-003` FastAPI loads persistent runtime state at startup.
- `AC-MVP-004` Multipart prediction returns decision data and a decodable heatmap.
- `AC-MVP-005` The backend validates, maps, and exposes stable errors.
- `AC-MVP-006` The desktop completes analysis and interactive heatmap display.
- `AC-MVP-007` Docker Compose reaches healthy inference and backend readiness.

### Evaluation and Calibration – Achieved Exploratorily

- `AC-EVA-001` An artifact can be evaluated from a labeled CSV without refitting.
- `AC-EVA-002` Invalid manifests and unsafe paths are rejected.
- `AC-EVA-003` Evaluation reports distributions, confusion data, metrics, group rates, and errors.
- `AC-EVA-004` The generalized workflow is exercised on VisA Candle.
- `AC-EVA-005` Schema version 2 records threshold provenance.
- `AC-EVA-006` Schema version 1 remains loadable.
- `AC-EVA-007` q100, q99, and q95 share identical feature memory.
- `AC-EVA-008` q95 is documented as provisional rather than final.

### Multi-Model Registry – Achieved

- `AC-REG-001` A registry with multiple enabled artifacts is validated during startup.
- `AC-REG-002` Invalid schemas, IDs, duplicates, defaults, and unsafe paths are rejected.
- `AC-REG-003` One persistent runtime loads for every enabled model.
- `AC-REG-004` The configured default is used when no model ID is supplied.
- `AC-REG-005` An explicit model ID selects the matching runtime.
- `AC-REG-006` Unknown model IDs return a clear failure.
- `AC-REG-007` `GET /api/v1/models` exposes enabled models and the default.
- `AC-REG-008` Legacy single-artifact startup remains compatible.
- `AC-REG-009` Backend and desktop consume the catalog without hard-coded model lists.
- `AC-REG-010` At least two models are selected successfully without recreating the service.
- `AC-REG-011` Responses identify the requested runtime.
- `AC-REG-012` The containerized stack loads the external registry and artifacts read-only.

### Independent Threshold Validation – Pending

- `AC-VAL-001` The q95 strategy is evaluated without further tuning on unused evidence.
- `AC-VAL-002` A strict calibration and independent final-test protocol is documented.
- `AC-VAL-003` Recommended normal-image counts are supported by evidence.

### Public Artifact Release – Pending

- `AC-REL-001` Redistribution rights are confirmed.
- `AC-REL-002` Checksums and provenance are published.
- `AC-REL-003` Results link to the exact artifact.
- `AC-REL-004` A Model Card documents intended use, limitations, and evaluation.

## Known Limitations and Risks

- artifacts depend on trusted PyTorch serialization;
- feature memories are large and exact search is compute-intensive;
- eager multi-model loading increases startup time and memory use;
- multiple service workers duplicate all feature memories;
- registry entries do not yet require artifact checksums;
- thresholds may be unstable for small or unrepresentative validation sets;
- provisional q95 evidence is not independently validated;
- direct square resizing may distort non-square products;
- metadata does not fully describe preprocessing or pretrained weights;
- dynamic registry reload and lazy loading are not implemented;
- additional cross-category model validation is required;
- heatmaps lack pixel-level quantitative validation;
- Base64 heatmaps increase response size;
- authentication and production hardening are incomplete;
- dataset licenses constrain redistribution and commercial use;
- the system is not certified for production decisions.

## Open Decisions

- What minimum normal-image count is useful for different categories?
- How should small datasets calibrate thresholds safely?
- How should non-square products be resized or padded?
- Which provenance and checksum fields become mandatory?
- Which memory-reduction method should be evaluated first?
- Should future runtimes load lazily or continue eager startup loading?
- How should a registry be reloaded safely without inconsistent requests?
- Which pixel-level localization metrics should be adopted?

## Delivery Stages

1. **Dataset and model foundation – completed.**
2. **Bottle and Capsule exploratory evaluation – completed.**
3. **Python artifact and inference MVP – completed.**
4. **FastAPI, ASP.NET Core, WPF, and Docker integration – completed.**
5. **General external-directory fitting – completed.**
6. **Dataset-independent artifact evaluation – completed.**
7. **VisA Candle threshold calibration – completed exploratorily.**
8. **Registry-based multi-model service – completed.**
9. **Backend, desktop, and stack multi-model integration – completed locally.**
10. **Registry-capable component releases – next.**
11. **Independent fixed-threshold validation – planned.**
12. **Production and public artifact hardening – future.**

## Related Documentation

- `ArchitectureOverview.md` – system boundaries and runtime flow
- `DatasetDocumentation.md` – dataset sources, licenses, inventories, and validation
- `DevelopmentStatus.md` – verified implementation status
- `ModelDevelopmentStrategy.md` – fitting, validation, evaluation, and approval rules
- `experiments/visa-candle-threshold-calibration.md` – exploratory calibration record
- a future `ModelCard.md` – released artifact behavior and limitations

## Last Updated

2026-08-21
