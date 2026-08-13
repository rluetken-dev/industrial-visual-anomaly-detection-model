# Industrial Visual Anomaly Detection – Architecture Overview

## Purpose

This document describes the current and intended architecture of the Industrial Visual Anomaly Detection system.

The architecture separates:

- local dataset storage and validation;
- Python-based model development and evaluation;
- versioned model artifacts;
- a future .NET inference backend;
- future web and desktop clients.

The project is currently in the dataset-preparation and anomaly-baseline preparation phase. Only components explicitly identified as verified or implemented currently exist. Backend and client components describe the intended target architecture and have not yet been created.

This is a living architecture document. It must be updated whenever technical evidence changes the selected model, preprocessing contract, artifact contract, repository boundaries, runtime responsibilities, or deployment strategy.

## Architecture Status

The document uses three status categories:

- **Verified** – demonstrated successfully through executable code or dataset validation;
- **Current direction** – selected for the first baseline or preferred target design, but not yet implemented completely;
- **Open** – requires further implementation, evaluation, or interoperability testing.

### Verified

- Python 3.12 and CPU-based PyTorch/TorchVision execution are operational.
- Pretrained ResNet18 weights can be loaded and executed locally.
- Intermediate `layer2` and `layer3` feature maps can be extracted.
- Multi-scale feature maps can be combined into local patch embeddings.
- A feature-extractor wrapper can be exported to ONNX.
- The exported ONNX model passes structural validation.
- ONNX Runtime reproduces the PyTorch feature outputs with very small numerical differences.
- Local CPU performance is sufficient for continued proof-of-concept development.
- MVTec AD, MVTec LOCO AD, and MVTec AD 2 have been downloaded, extracted, inspected, and validated.
- Direct 224 × 224 Bottle preprocessing has been verified technically and visually on normal and anomalous images.
- Direct resizing was selected over the default center-crop pipeline to preserve the complete bottle boundary.
- MVTec AD `bottle` has been selected as the first implementation category.
- The 209 normal bottle training images have been split deterministically into 167 fitting and 42 normal validation images using seed `42`.
- The split manifest is versioned and contains no overlap.

### Current Direction

- A PatchCore-style pipeline is the selected first anomaly-detection baseline.
- The first baseline uses MVTec AD `bottle`.
- The first feature extractor uses pretrained ResNet18 `layer2` and `layer3` outputs.
- The initial input size is 224 × 224 pixels.
- The pretrained backbone remains frozen for the first baseline.
- Python owns model development, fitting, threshold selection, evaluation, and artifact export.
- ONNX represents the neural feature extractor for later cross-runtime use.
- .NET is intended to provide production-oriented inference and application services.
- ASP.NET Core is intended to expose one client-neutral API.
- Future React and WPF clients should consume the same backend contract.
- Model, backend, web, and desktop concerns should remain independently deployable and separately versioned where practical.

### Open

- preprocessing strategies for later non-square categories;
- final patch-embedding aggregation details;
- whether the first Memory Bank uses all embeddings or a reduced coreset;
- coreset selection algorithm and sampling ratio;
- Python nearest-neighbor implementation;
- anomaly-map resizing and smoothing;
- image-level score aggregation;
- normal-validation threshold method;
- final MVP metric set;
- Memory Bank serialization format;
- division of scoring responsibilities between Python artifacts, ONNX, and .NET;
- persistence model;
- authentication requirements;
- artifact distribution and release strategy.

## Architectural Goals

The architecture should:

- keep model research independent from presentation technology;
- keep datasets and generated artifacts outside source control;
- make dataset splits and experiments reproducible;
- use the same preprocessing semantics during fitting and inference;
- support anomaly detection and localization;
- expose stable, client-neutral inspection results;
- allow desktop and web clients to share one backend contract;
- keep CPU inference viable;
- make model artifacts traceable and versioned;
- prevent test data from influencing fitting or threshold selection;
- support future model replacement without rewriting clients;
- clearly distinguish experimental software from a validated industrial quality-control system.

## System Context

The intended system context is:

```text
External datasets
        ↓
Python model-development repository
        ↓
Versioned model artifact package
        ↓
.NET backend and inference runtime
        ↓
Client-neutral API
        ↓
React web client      WPF desktop client
```

The current repository implements only the Python model-development portion. It produces experimental feature-extractor artifacts and will later produce evaluated anomaly-model artifacts. The future backend will consume released artifacts and expose inspection operations. Clients will submit images and render backend results.

## Repository Boundaries

The preferred long-term repository separation is:

```text
industrial-visual-anomaly-detection-model
industrial-visual-anomaly-detection-backend
industrial-visual-anomaly-detection-web
industrial-visual-anomaly-detection-desktop
```

Only the model repository currently exists.

### Model Repository

The current model repository owns:

- dataset documentation;
- dataset structure, inventory, readability, and mask validation;
- deterministic split creation;
- preprocessing experiments;
- pretrained feature extraction;
- anomaly-model development;
- threshold selection;
- evaluation;
- ONNX export;
- Python/ONNX parity checks;
- experiment configuration;
- artifact metadata preparation.

It must not own:

- ASP.NET Core hosting;
- WPF presentation code;
- React presentation code;
- client-specific workflows;
- production authentication;
- operational database persistence.

### Backend Repository

The future backend repository is intended to own:

- model-artifact discovery and validation;
- ONNX Runtime integration;
- production image preprocessing;
- Memory Bank loading;
- nearest-neighbor scoring;
- threshold application;
- inspection orchestration;
- stable request and response contracts;
- optional inspection-history persistence;
- health, diagnostics, and observability;
- backend tests.

The backend must not contain client-specific layout, colors, dialogs, or navigation behavior.

### Web Repository

The future React web repository is intended to own:

- image upload and preview;
- inspection submission;
- result visualization;
- anomaly-map overlays;
- inspection-history views;
- responsive web behavior;
- frontend validation and error presentation;
- web tests.

The web client must not duplicate model preprocessing or anomaly-scoring rules.

### Desktop Repository

The future WPF desktop repository is intended to own:

- desktop image selection;
- inspection submission;
- result and heatmap presentation;
- inspection-history views;
- desktop-specific commands and dialogs;
- WPF and ViewModel tests.

The first desktop version is expected to consume the backend API. Optional offline inference may be evaluated later but is not part of the initial architecture baseline.

## Current Model-Development Repository

The current repository contains:

```text
configs/
  splits/
    mvtec-ad-bottle-seed-42.json
docs/
  ArchitectureOverview.md
  DatasetDocumentation.md
  DevelopmentStatus.md
  ModelDevelopmentStrategy.md
  ProjectSpecification.md
scripts/
  create_mvtec_ad_split.py
  inspect_preprocessing.py
  validate_mvtec_ad.py
  validate_mvtec_ad_2.py
  validate_mvtec_loco_ad.py
.gitignore
.python-version
environment_check.py
README.md
requirements.txt
```

Local virtual environments, editor settings, datasets, generated reports, caches, ONNX exports, model artifacts, and experiment output are excluded from version control.

## Model Development Architecture

The selected first Python model pipeline is:

```text
MVTec AD bottle configuration
        ↓
Dataset validation
        ↓
Deterministic fitting/validation split
        ↓
Image preprocessing
        ↓
Pretrained ResNet18 feature extractor
        ↓
layer2 and layer3 feature maps
        ↓
Multi-scale local patch embeddings
        ↓
Optional coreset selection
        ↓
Normal feature Memory Bank
        ↓
Nearest-neighbor anomaly scoring
        ↓
Normal-validation threshold selection
        ↓
Official bottle test evaluation
        ↓
Versioned artifact export
```

### Dataset Configuration

Dataset configuration must define:

- local dataset root supplied at runtime;
- dataset family;
- category;
- fitting, validation, and test partitions;
- image and mask locations;
- permitted image extensions;
- preprocessing configuration;
- deterministic random seed.

Machine-specific absolute dataset paths must not be committed as public defaults. The deterministic membership of the first bottle split is stored in:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
```

### Dataset Validation

The implemented dataset validators currently verify relevant combinations of:

- expected categories and directories;
- required root files;
- partition inventories;
- readable PNG files;
- image dimensions and modes;
- image and mask counts;
- image-mask filename correspondence;
- mask dimensions, modes, and content.

The deterministic split was additionally checked for complete source coverage and zero overlap.

All three dataset validators can write schema-versioned JSON reports after successful validation. Generated reports currently contain resolved local dataset paths and remain ignored local output. Future validation may add duplicate-content detection. Validation must run before fitting or evaluation.

### Dataset Storage Boundary

Dataset archives and extracted data are stored outside the repository under:

```text
C:/dev/data/industrial-visual-anomaly-detection/
```

The repository contains documentation, checksums, scripts, and split manifests, but not the datasets themselves.

### Preprocessing

Fitting and runtime preprocessing must be equivalent.

The selected preprocessing contract for the initial MVTec AD Bottle baseline performs:

- image decoding;
- RGB conversion;
- direct resizing to 224 × 224 pixels;
- bilinear interpolation with antialiasing;
- conversion to a floating-point tensor;
- ImageNet normalization using the mean and standard deviation expected by the pretrained ResNet18 weights.

The resulting tensor shape is:

```text
(3, 224, 224)
```

The default TorchVision pipeline, which resizes to 256 × 256 pixels and then applies a 224 × 224 center crop, was evaluated visually as an alternative.

Direct resizing was selected because it preserves the complete bottle boundary and surrounding background. Center cropping removes part of the outer image area and could therefore discard defects near the object boundary.

All inspected MVTec AD Bottle images are square at 900 × 900 pixels. Direct resizing therefore introduces no aspect-ratio distortion for the selected first category.

The inspection script retains both preprocessing variants for visual comparison, but only direct 224 × 224 resizing is selected for the initial Bottle model pipeline.

Every preprocessing decision required for inference must later be represented in exported metadata and reproduced by the .NET runtime.

### Feature Extractor

The selected initial feature extractor is a pretrained ResNet18.

For an input tensor shaped `(1, 3, 224, 224)`, the verified outputs are:

```text
layer2: (1, 128, 28, 28)
layer3: (1, 256, 14, 14)
```

The `layer3` output is resized to 28 × 28 and concatenated with `layer2`, producing:

```text
(1, 384, 28, 28)
```

This is rearranged into:

```text
(784, 384)
```

Each input image therefore produces 784 local patch embeddings with 384 feature values each.

The backbone remains frozen for the first baseline. Fine-tuning is deferred until the frozen-feature baseline has been evaluated.

### Memory Bank

The planned Memory Bank represents local features extracted from the 167 normal bottle fitting images.

The first implementation must establish:

- deterministic embedding extraction order;
- tensor shape and data type;
- storage size before reduction;
- whether all embeddings can be retained practically;
- whether a coreset is required;
- serialization format;
- metadata linking the Memory Bank to the backbone and preprocessing version.

The Memory Bank must be treated as part of the model artifact. It must never be mixed across incompatible feature extractors, layers, input sizes, or normalization rules.

No Memory Bank has been built yet.

### Anomaly Scoring

The planned scoring process is:

```text
input image
→ preprocessing
→ patch embeddings
→ nearest-neighbor distances to normal Memory Bank
→ patch anomaly scores
→ image anomaly score
→ resized anomaly map
→ threshold-based decision
```

The exact nearest-neighbor implementation, image-score aggregation, map interpolation, smoothing, and threshold method remain open.

No anomaly-scoring implementation exists yet.

### Evaluation

The first evaluation uses the complete official MVTec AD bottle test partition and its supplied masks.

Evaluation must distinguish:

- image-level anomaly detection;
- pixel-level anomaly localization;
- runtime performance;
- artifact size and memory usage.

Candidate metrics include:

- image-level AUROC;
- pixel-level AUROC;
- Average Precision;
- Precision;
- Recall;
- F1 score;
- confusion matrix;
- inference time.

The exact MVP metric set remains open. Official test data must remain isolated from model fitting, threshold optimization, and hyperparameter selection. The 42 normal validation images are reserved for threshold selection and normal-score analysis.

No evaluation results exist yet.

## Model Artifact Architecture

The model repository is expected to export a versioned artifact package after the first model has been evaluated.

Candidate contents are:

```text
model-package/
├── feature-extractor.onnx
├── feature-memory.*
├── model-metadata.json
├── preprocessing.json
├── thresholds.json
├── evaluation-summary.json
├── checksums.txt
└── notices/
```

### Feature Extractor

`feature-extractor.onnx` contains the neural feature-extraction graph and pretrained parameters required for inference.

The current ONNX export is provisional. It proves export and numerical parity but is not a released model artifact.

### Feature Memory

The feature-memory artifact will contain the selected normal reference embeddings or coreset. Its serialization format has not yet been selected.

### Model Metadata

The metadata must identify at least:

- artifact schema version;
- model version;
- dataset and category;
- dataset archive checksum where appropriate;
- split-manifest reference;
- backbone and pretrained weight version;
- input dimensions;
- preprocessing configuration;
- selected feature layers;
- feature tensor dimensions;
- Memory Bank or coreset configuration;
- scoring configuration;
- threshold method and value;
- evaluation summary;
- framework versions;
- license and attribution information.

The backend must reject artifact packages with unsupported schema versions, missing required files, incompatible dimensions, or invalid checksums where checksums are required.

## .NET Runtime Architecture

The planned .NET runtime isolates model-specific behavior behind application-facing interfaces.

```text
API
 ↓
Application orchestration
 ↓
Inspection domain model
 ↓
Model runtime abstraction
 ↓
ONNX Runtime + feature memory + scoring
```

This runtime does not exist yet.

### Domain

The future domain layer should represent concepts such as:

- inspection request;
- inspection result;
- anomaly decision;
- anomaly score;
- anomaly region or heatmap reference;
- model identity;
- validation failure.

It must not reference ASP.NET Core, WPF, React, ONNX Runtime, or a database provider.

### Application

The future application layer should coordinate:

- request validation;
- image decoding;
- preprocessing;
- model selection;
- inference;
- anomaly scoring;
- persistence where enabled;
- response creation.

### Model Runtime

The future model runtime should own:

- artifact loading;
- ONNX Runtime session creation;
- feature-output validation;
- Memory Bank loading;
- nearest-neighbor search;
- patch-score aggregation;
- anomaly-map production;
- threshold application.

It should be independently testable without hosting the web API.

### Infrastructure

The future infrastructure layer may provide:

- file-system artifact storage;
- database-backed inspection history;
- image storage;
- clock and identifier services;
- logging adapters;
- configuration providers.

### API

The future ASP.NET Core API should provide client-neutral operations such as:

- inspect one image;
- retrieve model information;
- retrieve inspection history where persistence is enabled;
- report health and readiness.

The exact API contract is not yet implemented.

## Inspection Runtime Flow

The planned request flow is:

```text
Client submits image
        ↓
API validates request metadata and payload limits
        ↓
Application selects a compatible model artifact
        ↓
Runtime decodes and preprocesses the image
        ↓
ONNX feature extractor produces feature maps
        ↓
Runtime produces patch embeddings
        ↓
Nearest-neighbor search compares patches with normal memory
        ↓
Runtime builds patch scores and anomaly map
        ↓
Threshold produces normal/anomalous decision
        ↓
Application creates client-neutral result
        ↓
Optional persistence records inspection metadata
        ↓
Client renders score, decision, and overlay
```

## Client-Neutral Result Contract

The backend should return analysis information rather than presentation instructions.

Appropriate result fields may include:

- inspection identifier;
- model identifier and version;
- category;
- anomaly score;
- threshold;
- normal/anomalous decision;
- anomaly-map or overlay reference;
- processing duration;
- warnings;
- validation messages.

The backend should not return presentation-specific fields such as button colors, tab selection, dialog text, WPF commands, or React component names.

The exact schema remains illustrative until implemented.

## Persistence Direction

Inspection-history persistence is planned but not yet designed.

Potential persisted information includes:

- inspection identifier;
- timestamp;
- model and artifact versions;
- image reference or checksum;
- category;
- anomaly score and threshold;
- decision;
- processing duration;
- validation or runtime failures.

Raw image persistence must be optional and governed by privacy, retention, storage, and security requirements.

## Configuration Direction

Configuration should separate:

- public repository defaults;
- local development overrides;
- deployment environment values;
- secrets.

Machine-specific dataset roots, credentials, private endpoints, and secrets must not be committed.

Model behavior that affects reproducibility must be stored in versioned experiment or artifact metadata rather than only in local environment configuration.

## Security and Safety Considerations

The future backend must treat uploaded images and model artifacts as untrusted inputs.

Planned safeguards include:

- media-type and file-signature validation;
- bounded upload sizes;
- bounded decoded image dimensions;
- controlled temporary storage;
- request timeouts and cancellation;
- artifact schema and checksum validation;
- safe error responses;
- separation of public configuration and secrets.

The system must not be represented as a validated production quality-control system without appropriate domain validation, regulatory work, operational monitoring, and controlled deployment.

## Testing Strategy

### Python Tests

Python tests should cover:

- dataset structure and inventory validation;
- image and mask correspondence;
- deterministic split behavior;
- split overlap prevention;
- preprocessing output shape and values;
- feature-output dimensions;
- embedding construction;
- Memory Bank creation;
- anomaly scoring;
- threshold selection;
- metric calculation;
- artifact metadata validation;
- ONNX export and loading;
- PyTorch/ONNX numerical parity.

### .NET Runtime Tests

Future .NET tests should cover:

- artifact compatibility checks;
- preprocessing parity with Python;
- ONNX input and output dimensions;
- feature-memory loading;
- nearest-neighbor scoring;
- threshold application;
- malformed image handling;
- cancellation and timeout behavior.

### API Tests

Future API tests should cover:

- valid inspection requests;
- unsupported media types;
- oversized payloads;
- invalid model identifiers;
- unavailable artifacts;
- health and readiness behavior;
- stable error contracts.

### Client Tests

Future client tests should cover:

- image selection;
- request submission;
- loading, success, and error states;
- result rendering;
- anomaly-map overlays;
- validation messages.

### Cross-Runtime Contract Tests

Cross-runtime tests should compare Python and .NET results for fixed fixtures, including:

- preprocessed tensors;
- ONNX feature outputs;
- patch embeddings;
- nearest-neighbor distances;
- image-level scores;
- anomaly-map dimensions;
- threshold decisions.

## Observability Direction

The future backend should record:

- model and artifact versions;
- inference duration;
- preprocessing duration;
- scoring duration;
- validation failures;
- artifact-loading failures;
- request outcome;
- resource usage where practical.

Logs must not expose secrets or sensitive image contents.

## Deployment Direction

Initial deployment should remain simple and CPU-compatible.

Potential future modes include:

- local development with Python tooling;
- a self-contained .NET backend using ONNX Runtime;
- a centrally hosted web API;
- a WPF client consuming that API;
- a React client consuming that API;
- optional offline desktop inference after parity is proven.

Deployment design must follow evaluated model behavior rather than being finalized before the baseline exists.

## Known Architectural Risks

- A full PatchCore Memory Bank may become large.
- CPU nearest-neighbor search may become a bottleneck.
- Incorrect preprocessing parity can invalidate cross-runtime results.
- Direct resizing to 224 × 224 may reduce the visibility of very small defects.
- Later non-square categories may require padding, aspect-ratio-preserving resizing, or category-specific preprocessing.
- Threshold selection can leak test information if validation is not isolated.
- ONNX may not represent the complete anomaly pipeline conveniently.
- Model updates may break older runtimes without artifact schema versioning.
- Dataset licenses constrain redistribution and commercial use.
- Public benchmark performance may not transfer to real industrial images.
- A prototype may be misunderstood as a validated inspection system.

## Current Non-Goals

The current phase does not include:

- production deployment;
- real-time camera integration;
- PLC integration;
- user authentication;
- a production database;
- web or desktop UI implementation;
- supervised defect-type classification;
- multi-model orchestration;
- automated retraining;
- regulatory validation;
- real pharmaceutical packaging validation.

## Completed Architectural Validation Steps

The following steps have been completed:

1. Establish a reproducible Python environment.
2. Verify pretrained ResNet18 execution on CPU.
3. Verify intermediate feature extraction and patch-embedding construction.
4. Export the feature extractor to ONNX.
5. Verify PyTorch and ONNX Runtime numerical parity.
6. Acquire and document MVTec AD, MVTec LOCO AD, and MVTec AD 2.
7. Validate dataset structures, inventories, image readability, and available masks.
8. Select MVTec AD bottle as the first category.
9. Create and verify the deterministic 167/42 bottle split.
10. Compare the default center-crop pipeline with direct 224 × 224 resizing on normal and anomalous Bottle images.
11. Add schema-versioned JSON reporting to all three dataset validators.
12. Correct MVTec LOCO AD mask validation to use category-specific values from `defects_config.json`.
13. Select direct 224 × 224 resizing as the initial Bottle preprocessing contract.

## Immediate Architectural Validation Steps

The next steps should be performed in this order:

1. Refactor the Python technical spike into reusable, testable modules.
2. Load the Bottle partitions through the deterministic split manifest.
3. Add automated tests for dataset report generation and schema contents.
4. Build and measure the first normal feature memory.
5. Implement and verify nearest-neighbor anomaly scoring.
6. Select a threshold using only normal validation scores.
7. Evaluate the baseline on the official Bottle test partition.
8. Define the first artifact metadata schema from the evaluated implementation.
9. Package the ONNX model, feature memory, metadata, threshold, and evaluation summary.
10. Create a minimal .NET console spike that loads the package.
11. Verify Python/.NET preprocessing and scoring parity.
12. Finalize backend repository structure and implement the ASP.NET Core API.
13. Add web and desktop clients only after the backend contract is stable.

## Related Documentation

- `DevelopmentStatus.md` records verified results and the current implementation state.
- `ProjectSpecification.md` defines the current product scope and requirements.
- `ModelDevelopmentStrategy.md` defines fitting, validation, evaluation, and experiment rules.
- `DatasetDocumentation.md` records dataset sources, licenses, structures, validation results, and the selected initial category.
- A future `ModelCard.md` will document a specific evaluated model artifact after one exists.

## Last Updated

This architecture reflects the verified project state as of 2026-08-13.
