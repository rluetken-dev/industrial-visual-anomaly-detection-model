# Industrial Visual Anomaly Detection – Project Specification

## Document Purpose

This document defines the current functional and non-functional requirements for the Industrial Visual Anomaly Detection project.

It distinguishes between:

- the implemented Python model-development and inference MVP;
- requirements that are partially implemented or still experimental;
- the planned backend, web, and desktop system.

Implementation history and short-term work are recorded in `DevelopmentStatus.md`. Architectural responsibilities and boundaries are described in `ArchitectureOverview.md`.

## Requirement Identification

Requirement identifiers use the following prefixes:

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
- `API` – Backend API
- `WEB` – Web Application
- `DSK` – Desktop Application
- `OBS` – Observability and Diagnostics
- `SEC` – Security and Privacy

Identifiers follow `<type>-<area>-<number>`, for example `FR-INF-001`.

## Project Overview

Industrial Visual Anomaly Detection is a portfolio-oriented computer-vision system for detecting unusual visual patterns in industrial inspection images and highlighting suspicious image regions.

The implemented Python MVP uses a PatchCore-inspired method:

1. normal images are preprocessed deterministically;
2. a frozen pretrained ResNet18 extracts intermediate features;
3. `layer2` and `layer3` features are combined into local patch embeddings;
4. embeddings from normal fitting images form a feature memory;
5. query patches are compared with their nearest normal neighbor;
6. patch distances form an anomaly map;
7. an aggregation rule produces one image-level anomaly score;
8. a threshold derived only from held-out normal validation images determines the final decision.

The system detects deviation from learned normal appearance. It does not currently classify the exact defect type.

## Current Implementation Status

### Implemented and Verified

- Python 3.12 CPU development environment;
- reusable `src`-layout Python package;
- deterministic RGB preprocessing with direct square resize and ImageNet normalization;
- configurable input resolution;
- frozen pretrained ResNet18 feature extraction from `layer2` and `layer3`;
- 384-dimensional multi-scale patch embeddings;
- complete normal feature-memory construction;
- deterministic random feature-memory sampling as an experimental option;
- exact chunked Euclidean nearest-neighbor search;
- patch-level anomaly maps;
- maximum and top-fraction-mean image-score aggregation;
- normal-validation-based threshold selection;
- MVTec AD category discovery and labeled test loading;
- image-level classification metrics and per-group reporting;
- anomaly heatmap generation and overlays;
- versioned PyTorch model artifacts containing metadata and feature memory;
- artifact validation, writing, and loading;
- single-image Python inference API and command-line interface;
- export command for reusable category-specific artifacts;
- automated unit tests for the main dataset, model, evaluation, visualization, sampling, and artifact components.

### Verified Datasets

The local copies of MVTec AD, MVTec LOCO AD, and MVTec AD 2 have passed structural, inventory, image-readability, and available mask validations. Machine-readable JSON validation reports can be generated locally.

### Reference Categories

`bottle` was the first baseline category. It uses a deterministic split of 209 normal training images into 167 fitting and 42 validation images.

`capsule` is the current artifact and inference reference category. It uses a deterministic split of 219 normal training images into 175 fitting and 44 validation images.

The selected Capsule reference configuration is:

- input size: 320 × 320;
- patch grid: 40 × 40;
- embedding dimension: 384;
- image-score aggregation: mean of the highest-scoring 1% of patches;
- threshold source: maximum normal validation score;
- feature memory: complete, unsampled fitting memory;
- feature-memory shape: `(280000, 384)`;
- feature-memory size: approximately 410.16 MiB;
- threshold: approximately `2.501822`.

### Not Yet Implemented

- pixel-level benchmark metrics against ground-truth masks;
- systematic multi-category benchmark execution;
- framework-neutral production artifact format;
- updated ONNX export and parity verification for the selected 320 × 320 pipeline;
- ASP.NET Core backend;
- web application;
- desktop application;
- production authentication, authorization, persistence, and observability.

## Project Goals

The project shall demonstrate:

- practical reuse of pretrained computer-vision features;
- normal-only anomaly fitting without defect examples;
- reproducible image-level anomaly detection;
- understandable spatial anomaly visualization;
- deterministic evaluation and artifact generation;
- a reusable model boundary suitable for later application integration;
- a future client-neutral backend serving both web and desktop clients;
- transparent reporting of assumptions, limitations, and exploratory results.

## Intended Users

- developers learning computer vision and anomaly detection;
- engineers evaluating visual inspection workflows;
- portfolio reviewers examining architecture, testing, and reproducibility;
- future web and desktop users submitting inspection images and reviewing results.

The project is not currently certified for production quality control or safety-critical decisions.

## Scope

### Current Python MVP

The current scope includes dataset qualification, deterministic splitting, preprocessing, feature-memory creation, anomaly scoring, evaluation, visualization, artifact export, artifact loading, and single-image inference.

### Planned Application Scope

The planned system adds an ASP.NET Core backend and separate web and desktop clients. Both clients shall consume the same backend contract rather than embedding independent business logic.

### Explicitly Deferred

- exact defect-type classification;
- supervised detector training;
- continuous video inspection;
- camera and programmable-logic-controller integration;
- automated model retraining in production;
- multi-tenant operation;
- safety certification;
- commercial use of datasets without separate license review.

## Functional Requirements

### Dataset Management

- `FR-DAT-001` The system shall accept dataset roots outside the Git repository.
- `FR-DAT-002` Dataset validators shall verify expected structure, inventory, readable images, and available mask relationships.
- `FR-DAT-003` Dataset validators shall support optional schema-versioned JSON reports.
- `FR-DAT-004` Raw datasets and generated validation reports shall remain excluded from Git.
- `FR-DAT-005` A deterministic split manifest shall identify fitting and normal-validation images.
- `FR-DAT-006` Split validation shall reject duplicate, overlapping, absolute, or parent-traversing image paths.
- `FR-DAT-007` Dataset paths resolved from a manifest shall be verified to exist before model processing.
- `FR-DAT-008` MVTec AD test discovery shall preserve category, defect group, and normal/anomalous labels.

### Image Preprocessing

- `FR-PRE-001` Input images shall be converted to RGB.
- `FR-PRE-002` Images shall be resized directly to the configured square input size using bilinear interpolation with antialiasing.
- `FR-PRE-003` Images shall be converted to floating-point tensors and normalized with ImageNet mean and standard deviation.
- `FR-PRE-004` Fitting, validation, evaluation, export, and inference shall use the same preprocessing definition for a given model configuration.
- `FR-PRE-005` The selected input size shall be recorded in exported artifact metadata.
- `FR-PRE-006` Preprocessing inspection tooling shall permit visual verification of resizing decisions.

### Model Development

- `FR-MOD-001` The first baseline shall use a pretrained ResNet18 with frozen parameters.
- `FR-MOD-002` The feature extractor shall expose `layer2` and `layer3` feature maps.
- `FR-MOD-003` The embedding pipeline shall resize `layer3` features to the `layer2` spatial grid and concatenate both channel dimensions.
- `FR-MOD-004` Each patch embedding shall contain 384 values.
- `FR-MOD-005` Feature memory shall be built only from normal fitting images.
- `FR-MOD-006` Complete feature memory shall remain the reference behavior.
- `FR-MOD-007` Optional memory sampling shall be deterministic for a fixed sampling seed.
- `FR-MOD-008` Sampling experiments shall report retained fraction and resulting memory size.

### Anomaly Inference

- `FR-INF-001` Each query patch shall be scored by its exact Euclidean distance to the nearest feature-memory entry.
- `FR-INF-002` Nearest-neighbor computation shall support chunking to limit temporary memory consumption.
- `FR-INF-003` Patch scores shall be reconstructed into a spatial grid for every image.
- `FR-INF-004` Image scores shall support maximum aggregation and top-fraction-mean aggregation.
- `FR-INF-005` A prediction shall be anomalous only when its score is strictly greater than the configured threshold.
- `FR-INF-006` Single-image inference shall return image path, anomaly score, threshold, decision, and patch-score grid.
- `FR-INF-007` Inference shall derive input size, patch-grid size, aggregation, and threshold from the loaded artifact.
- `FR-INF-008` The Python CLI shall display the artifact configuration, score, threshold, decision, and relevant timings.

### Evaluation

- `FR-EVA-001` The decision threshold shall be derived only from held-out normal validation images.
- `FR-EVA-002` The current threshold rule shall use the maximum normal validation image score.
- `FR-EVA-003` Evaluation shall report score distributions for normal validation data and each test defect group.
- `FR-EVA-004` Evaluation shall report true positives, true negatives, false positives, false negatives, accuracy, precision, recall, and F1 score.
- `FR-EVA-005` Evaluation shall report detection rates by test group and list false negatives.
- `FR-EVA-006` Aggregation comparisons shall reuse the same patch-score tensors when possible.
- `FR-EVA-007` Results obtained after inspecting the test partition shall be identified as exploratory rather than unbiased final estimates.
- `FR-EVA-008` Future pixel-level evaluation shall compare anomaly maps with ground-truth masks using explicitly selected metrics.

### Visualization

- `FR-VIS-001` The system shall resize a patch-score grid to image resolution using bilinear interpolation.
- `FR-VIS-002` Anomaly maps shall support per-image normalization for visual inspection.
- `FR-VIS-003` Anomaly maps shall support fixed threshold-based normalization for cross-image visual comparison.
- `FR-VIS-004` The system shall create RGB heatmaps and configurable-opacity overlays.
- `FR-VIS-005` Heatmaps shall remain an explanation aid and shall not replace the image-level decision contract.

### Model Artifacts

- `FR-ART-001` A model artifact shall contain `metadata.json` and `feature_memory.pt`.
- `FR-ART-002` Artifact metadata shall include schema version, dataset, category, backbone, input size, patch-grid size, embedding dimension, aggregation method, top fraction, threshold, memory fraction, sampling seed, and feature-memory entry count.
- `FR-ART-003` The artifact writer shall reject empty, non-finite, incorrectly shaped, or metadata-inconsistent feature memory.
- `FR-ART-004` The artifact loader shall validate required files, typed metadata, tensor dimensionality, finite values, entry count, and embedding dimension.
- `FR-ART-005` Artifact tensors shall load onto CPU.
- `FR-ART-006` Generated model artifacts shall remain excluded from Git.
- `FR-ART-007` The export command shall rebuild fitting memory, calculate the normal-validation threshold, and write one reusable artifact.
- `FR-ART-008` A future production artifact shall eliminate unnecessary Python/PyTorch runtime coupling or explicitly version that dependency.
- `FR-ART-009` A future artifact schema shall contain a complete, self-describing preprocessing contract rather than only input size.

### Backend API – Planned

- `FR-API-001` A future ASP.NET Core backend shall own application-level inference orchestration.
- `FR-API-002` The backend shall expose a versioned endpoint for submitting an image and receiving an anomaly result.
- `FR-API-003` The result contract shall include score, threshold, decision, model identity, and optional localization data.
- `FR-API-004` The backend shall validate file type, size, decoding, and request limits.
- `FR-API-005` Web and desktop clients shall consume the same backend contract.
- `FR-API-006` The backend shall not silently change model artifacts or thresholds.

### Web Application – Planned

- `FR-WEB-001` The web client shall allow an image to be selected and submitted.
- `FR-WEB-002` The web client shall display the anomaly decision, score, threshold, and model identity.
- `FR-WEB-003` The web client shall display localization output when supplied by the backend.
- `FR-WEB-004` Errors shall be understandable and shall not expose sensitive server details.

### Desktop Application – Planned

- `FR-DSK-001` The desktop client shall allow local image selection and submission to the shared backend.
- `FR-DSK-002` The desktop client shall display the same essential result fields as the web client.
- `FR-DSK-003` Desktop-specific convenience features shall not create a separate inference contract.

### Observability and Diagnostics – Planned

- `FR-OBS-001` Inference diagnostics shall record model identity, duration, outcome, and failure category without logging raw image content by default.
- `FR-OBS-002` Health information shall distinguish application availability from model readiness.
- `FR-OBS-003` Operational logs shall use correlation identifiers for request tracing.

## Business Rules

- `BR-MOD-001` Only normal fitting images may populate the feature memory.
- `BR-EVA-001` Normal validation images may determine thresholds but may not populate feature memory.
- `BR-EVA-002` Benchmark test labels may be used for evaluation but not for fitting or threshold selection.
- `BR-INF-001` A score equal to the threshold is classified as normal; only a greater score is anomalous.
- `BR-ART-001` Each artifact is category-specific unless a future schema explicitly declares otherwise.
- `BR-ART-002` An artifact shall be treated as immutable after release.
- `BR-DAT-001` Dataset licensing shall be reviewed before any public or commercial distribution.

## Non-Functional Requirements

### Reproducibility

- `NFR-MOD-001` Deterministic operations shall record their seeds and configuration.
- `NFR-DAT-001` Versioned manifests shall make fitting and validation membership reproducible.
- `NFR-ART-001` Exported artifacts shall contain enough metadata to identify their model configuration.
- `NFR-EVA-001` Reported results shall state category, resolution, memory fraction, aggregation rule, and threshold rule.

### Performance

- `NFR-INF-001` The reference implementation shall support CPU-only inference.
- `NFR-INF-002` Distance computation shall avoid materializing the complete query-to-memory distance matrix.
- `NFR-INF-003` Performance measurements shall separate artifact loading, extractor creation, feature-memory construction, and scoring where relevant.
- `NFR-INF-004` Performance targets for a production backend shall be established only after the runtime and artifact format are selected.

### Compatibility and Portability

- `NFR-ART-002` Current PyTorch artifacts shall be labeled as Python-runtime artifacts, not runtime-neutral artifacts.
- `NFR-ART-003` A future .NET integration shall use a verified compatible artifact or service boundary.
- `NFR-MOD-002` ONNX parity shall be verified again for the selected production resolution before ONNX is used outside the technical spike.

### Maintainability and Testability

- `NFR-MOD-003` Reusable model logic shall reside in the Python package rather than only in scripts.
- `NFR-MOD-004` Public package functions shall validate invalid and inconsistent inputs with clear errors.
- `NFR-MOD-005` Core deterministic components shall have automated unit tests.
- `NFR-MOD-006` Command-line scripts shall compose reusable package components instead of duplicating model logic.

### Security and Privacy

- `NFR-SEC-001` Uploaded images shall be treated as untrusted input.
- `NFR-SEC-002` Future APIs shall enforce file-size, content-type, decoding, and request-time limits.
- `NFR-SEC-003` Raw inspection images shall not be retained or logged by default.
- `NFR-SEC-004` Model artifacts shall not be accepted from untrusted sources without integrity controls.
- `NFR-SEC-005` Secrets and environment-specific configuration shall not be committed to Git.

### Documentation

- `NFR-MOD-007` Documentation shall distinguish verified facts, exploratory measurements, selected decisions, and planned work.
- `NFR-MOD-008` Major architectural or model decisions shall be reflected in the living documentation before a release milestone.

## Acceptance Criteria

### Python Model MVP – Achieved

- `AC-MOD-001` A normal-only feature memory can be built from a deterministic manifest.
- `AC-MOD-002` Normal and anomalous test images can be scored without gradient tracking.
- `AC-INF-001` Patch scores and an image-level anomaly decision are produced.
- `AC-EVA-001` Image-level metrics and defect-group results are reported.
- `AC-VIS-001` Anomaly overlays can be generated for normal and anomalous images.
- `AC-MOD-003` Automated tests, compilation checks, and dependency checks pass.

### Artifact and Local Inference MVP – Achieved

- `AC-ART-001` A Capsule 320 × 320 reference artifact can be exported and loaded.
- `AC-ART-002` The loaded feature-memory shape and metadata match the exported configuration.
- `AC-INF-002` The CLI classifies a known normal Capsule image as normal.
- `AC-INF-003` The CLI classifies a known anomalous Capsule image as anomalous.
- `AC-INF-004` Single-image prediction returns a 40 × 40 patch-score grid.

### Model Evaluation Expansion – Pending

- `AC-EVA-002` At least one category beyond the current Bottle and Capsule references is evaluated through the same generic workflow without category-specific model code.
- `AC-EVA-003` Pixel-level localization metrics are implemented and verified against benchmark masks.
- `AC-EVA-004` Final evaluation procedures avoid using the test partition for configuration selection, or explicitly provide a new untouched holdout.

### Runtime-Portability Milestone – Pending

- `AC-ART-003` The selected 320 × 320 feature extractor is exported to the intended portable runtime format.
- `AC-ART-004` Numerical parity is verified using representative real images.
- `AC-ART-005` Preprocessing and supporting model data can be reconstructed outside the current Python codebase.

### Backend MVP – Pending

- `AC-API-001` A versioned backend endpoint accepts a valid image and returns the defined inference result.
- `AC-API-002` Invalid and oversized files are rejected safely.
- `AC-API-003` Backend results agree with the verified Python reference for fixed test images within defined tolerances.

### Client MVPs – Pending

- `AC-WEB-001` The web client submits an image and displays the backend result.
- `AC-DSK-001` The desktop client submits an image and displays the backend result.
- `AC-API-004` Both clients use the same versioned API contract.

## Known Reference Results

The following results are exploratory because MVTec AD test data was inspected during development.

### Bottle, 224 × 224, Complete Memory

- maximum aggregation: accuracy `0.9398`, precision `1.0000`, recall `0.9206`, F1 `0.9587`, 0 false positives, 5 false negatives;
- top 1% mean: accuracy, precision, recall, and F1 `1.0000`, with 0 false positives and 0 false negatives;
- top 5% mean: accuracy `0.9759`, precision `0.9692`, recall `1.0000`, F1 `0.9844`, 2 false positives, 0 false negatives.

### Capsule, 320 × 320, Complete Memory, Top 1% Mean

- true positives: 104;
- true negatives: 21;
- false positives: 2;
- false negatives: 5;
- accuracy: `0.9470`;
- precision: `0.9811`;
- recall: `0.9541`;
- F1: `0.9674`.

Random sampling reduced memory and runtime but also reduced Capsule recall. Complete memory therefore remains the selected reference configuration.

## Known Limitations and Risks

- Current artifacts depend on PyTorch tensor serialization.
- Artifact metadata records input size but does not yet fully describe all preprocessing operations and constants.
- The selected 320 × 320 pipeline has not yet received its final ONNX export and parity verification.
- Exact nearest-neighbor search over complete feature memory is computationally and memory intensive.
- The current threshold rule is simple and may not generalize equally across categories or deployment conditions.
- Test data has influenced exploratory model decisions, so current benchmark numbers are not unbiased final estimates.
- Heatmaps are visually useful but have not yet been validated with pixel-level metrics.
- A model fitted on one category shall not be assumed to work on another category.
- Lighting, alignment, camera, material, and production changes may cause false alarms or missed anomalies.
- MVTec dataset licenses restrict commercial use and redistribution.
- No backend or client security controls exist because those components have not been implemented.

## Open Decisions

- Which additional MVTec AD categories should form the next generalization benchmark?
- Which pixel-level metrics should be the required localization acceptance measures?
- Should the next optimization use a principled coreset method, an indexed nearest-neighbor library, or both?
- Should the production boundary use ONNX Runtime inside .NET or keep Python inference behind a service boundary?
- How should complete preprocessing metadata be represented in the next artifact schema?
- Which artifact integrity and version-compatibility controls are required?
- Which backend API result schema best represents heatmaps or localization output?
- Which web framework and desktop technology will be used?

## Delivery Stages

1. **Dataset and model foundation – completed:** validated datasets, deterministic splits, preprocessing, frozen feature extraction, patch embeddings, feature memory, scoring, and tests.
2. **Evaluation and visualization baseline – completed:** image-level metrics, aggregation comparison, group analysis, and heatmap overlays.
3. **Python artifact and inference MVP – completed:** versioned artifact, exporter, loader, inference API, and CLI.
4. **Model hardening – next:** broader category evaluation, pixel-level metrics, improved performance strategy, and final portable model export.
5. **Backend MVP – planned:** ASP.NET Core inference API and operational safeguards.
6. **Client MVPs – planned:** separate web and desktop clients using the shared API.
7. **Portfolio release – planned:** verified setup, documentation, demonstrations, and clearly scoped limitations.

## Related Documentation

- `README.md` – repository introduction and setup
- `DevelopmentStatus.md` – verified progress and current next steps
- `ArchitectureOverview.md` – components, responsibilities, and data flow
- `ModelDevelopmentStrategy.md` – experimentation and model-selection strategy
- `DatasetDocumentation.md` – dataset sources, licenses, inventories, and validation
- `COMMITS.md` – commit-message conventions

## Last Updated

2026-08-13
