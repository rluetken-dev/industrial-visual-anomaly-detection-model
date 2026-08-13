# Industrial Visual Anomaly Detection – Project Specification

## Document Purpose

This document defines the current functional and non-functional requirements for the Industrial Visual Anomaly Detection project.

The project is under active development. Requirements describe the intended target unless they are explicitly identified as verified, deferred, or open. Current implementation progress is recorded separately in `DevelopmentStatus.md`.

## Requirement Identification

### Requirement Types

- `FR` – Functional Requirement
- `BR` – Business Rule
- `NFR` – Non-Functional Requirement
- `AC` – Acceptance Criterion

### Functional Areas

- `DAT` – Dataset Management
- `PRE` – Image Preprocessing
- `MOD` – Model Development
- `INF` – Anomaly Inference
- `EVA` – Evaluation
- `ART` – Model Artifacts
- `API` – Backend API
- `WEB` – Web Application
- `DSK` – Desktop Application
- `OBS` – Observability and Diagnostics
- `APP` – Cross-Cutting Application Concerns
- `SEC` – Security and Privacy

Requirement identifiers follow this structure:

```text
<type>-<area>-<number>
```

Example:

```text
FR-DAT-001
```

Numbers are unique within each combination of requirement type and functional area. Existing identifiers must not be reused when requirements are removed or replaced.

## Project Overview

Industrial Visual Anomaly Detection is a portfolio-oriented computer-vision system for detecting and localizing unusual visual patterns in industrial inspection images.

The intended end-to-end system consists of:

1. dataset qualification and validation;
2. deterministic image preprocessing;
3. fitting and evaluating an anomaly-detection model in Python;
4. exporting versioned, runtime-neutral inference artifacts;
5. serving inference through a future ASP.NET Core backend;
6. consuming the backend from future web and desktop clients.

The first model-development cycle uses MVTec AD `bottle` and targets CPU-compatible inference. A PatchCore-style feature-memory approach with a frozen pretrained ResNet18 is the selected first anomaly baseline.

The first model detects deviation from learned normal appearance and localizes suspicious regions. It is not initially required to classify the exact defect type.

## Current Verified Foundation

The following foundation has already been verified:

- Python 3.12 CPU development environment;
- pretrained ResNet18 execution;
- extraction of `layer2` and `layer3` features;
- multi-scale patch-embedding construction;
- provisional ONNX export and structural validation;
- PyTorch/ONNX Runtime numerical parity on an artificial reference tensor;
- execution of the selected TorchVision preprocessing on a real Bottle image;
- acquisition and validation of MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- selection of MVTec AD Bottle as the first category;
- deterministic division of 209 normal Bottle images into 167 fitting and 42 normal validation images using seed `42`;
- zero overlap and complete coverage in the versioned split manifest.

No feature memory, anomaly scorer, threshold, final test evaluation, production artifact package, .NET backend, web client, or desktop client has yet been implemented.

## Project Goals

The project should demonstrate:

- practical use of pretrained computer-vision features;
- normal-only anomaly-model fitting;
- image-level anomaly detection;
- spatial anomaly localization using benchmark masks;
- reproducible experimentation and evaluation;
- portable inference artifacts using ONNX and framework-neutral supporting data;
- later integration between Python model development and .NET applications;
- a client-neutral backend design;
- transparent communication of model limitations.

## Intended Users

The intended users are:

- developers evaluating the architecture and implementation;
- technical reviewers exploring the portfolio project;
- maintainers running model-development and evaluation workflows;
- future users experimenting with supported images through a backend or client.

The project is not intended for unsupervised use by production operators or for safety-critical decisions.

## Delivery Stages

### Stage 1: Model-Development MVP

The immediate target includes:

- MVTec AD Bottle dataset configuration and validation;
- deterministic fitting and normal-validation partitions;
- deterministic image preprocessing;
- a PatchCore-style anomaly baseline;
- complete initial feature-memory measurement before optimization;
- normal-validation-based threshold selection;
- locked evaluation on the official Bottle test partition;
- image-level anomaly scores and decisions;
- spatial anomaly maps;
- image-level and pixel-level metrics;
- reproducible experiment records;
- provisional model-artifact export;
- automated tests and documentation appropriate to the implemented state.

### Stage 2: Artifact and Backend MVP

After the Python model baseline has been evaluated, the target expands to:

- a versioned model-artifact package;
- a stable metadata schema;
- Python/ONNX/.NET parity fixtures;
- .NET artifact loading and validation;
- CPU-compatible ONNX Runtime inference;
- an ASP.NET Core inference API;
- automated backend and contract tests.

### Stage 3: Client MVPs

After the backend contract is stable, the target may expand to:

- a React web client;
- a WPF desktop client;
- image selection and submission;
- anomaly decision and score presentation;
- anomaly-map visualization;
- loading, validation, inference-error, and unavailable-model states.

### Deferred From The Initial Project

- live camera integration;
- real-time production-line control;
- PLC integration;
- automatic rejection of physical products;
- continual or online learning;
- automated retraining;
- simultaneous optimization for all available dataset categories;
- supervised defect-type classification;
- distributed or multi-GPU training;
- regulatory validation;
- use of private pharmaceutical production images;
- deployment as a certified inspection system.

## System Context

The intended system consists of independently maintainable components:

- the current Python model-development repository;
- versioned model-artifact packages;
- a future ASP.NET Core backend;
- a future React web application;
- a future WPF desktop application.

The model repository is implemented first. Backend and client repositories should be created only when their preceding contracts are sufficiently stable.

Regardless of repository layout, clients must consume inference through defined contracts rather than duplicating model logic.

## Functional Requirements

### Dataset Management

- **FR-DAT-001:** The model-development application must accept a configured dataset root rather than relying on a committed developer-specific absolute path.
- **FR-DAT-002:** The application must validate that the configured dataset family and category exist before fitting or evaluation begins.
- **FR-DAT-003:** The application must enumerate supported image files deterministically.
- **FR-DAT-004:** The application must identify normal and anomalous samples according to the documented dataset structure.
- **FR-DAT-005:** The application must associate anomaly masks with their source images when the selected dataset provides masks.
- **FR-DAT-006:** The application must detect missing, unreadable, unsupported, or structurally inconsistent required dataset files.
- **FR-DAT-007:** The application must generate a machine-readable dataset inventory or validation report.
- **FR-DAT-008:** The application must preserve an isolated final test partition.
- **FR-DAT-009:** The application must support a reproducible validation strategy that does not tune against final test labels.
- **FR-DAT-010:** The first Bottle implementation must load fitting and validation membership from `configs/splits/mvtec-ad-bottle-seed-42.json`.
- **FR-DAT-011:** The split loader must reject missing entries, duplicate membership, split overlap, or references outside the configured normal source partition.
- **FR-DAT-012:** Dataset archives and extracted images must remain outside the source repository.

### Image Preprocessing

- **FR-PRE-001:** The system must convert supported source images into the tensor format expected by the selected model configuration.
- **FR-PRE-002:** Preprocessing must define image decoding, color conversion, resizing, cropping, interpolation, scaling, channel order, normalization, data type, and final tensor shape.
- **FR-PRE-003:** Fitting, validation, and final-evaluation preprocessing must be deterministic.
- **FR-PRE-004:** Equivalent input images must produce numerically compatible model inputs across validated Python and future .NET inference paths.
- **FR-PRE-005:** Optional augmentation must be configured separately from deterministic evaluation preprocessing.
- **FR-PRE-006:** Normal-data augmentation must not intentionally transform a normal sample into an apparent defect.
- **FR-PRE-007:** The first baseline must use the default pretrained ResNet18 transform with RGB conversion, resize to 256, 224 × 224 center crop, bilinear interpolation, `float32` tensor conversion, and the expected ImageNet normalization.
- **FR-PRE-008:** A visual preprocessing check must preserve or generate comparable original, resized, and cropped images before the preprocessing contract is finalized.

### Model Development

- **FR-MOD-001:** The first baseline must use ResNet18 with documented default pretrained TorchVision weights.
- **FR-MOD-002:** The first baseline must extract `layer2` and `layer3` feature maps.
- **FR-MOD-003:** The application must align and transform the selected feature maps into local patch embeddings.
- **FR-MOD-004:** The application must fit a normal anomaly representation using only the 167 approved fitting images.
- **FR-MOD-005:** The application must implement a PatchCore-style feature-memory anomaly baseline.
- **FR-MOD-006:** Model configuration must be externalized or recorded rather than existing only as undocumented source-code constants.
- **FR-MOD-007:** Each fitting run must produce or reference an identifiable experiment record.
- **FR-MOD-008:** Randomized fitting or sampling operations must support explicit seeds.
- **FR-MOD-009:** The complete feature memory must be measured before coreset reduction is treated as required optimization.
- **FR-MOD-010:** The pretrained backbone must remain frozen during the first baseline.
- **FR-MOD-011:** A simpler global-feature reference may be implemented later, but it is not required for completion of the first PatchCore-style baseline.
- **FR-MOD-012:** More complex variants must not replace the documented baseline without measured and recorded evidence of their value.

### Anomaly Inference

- **FR-INF-001:** The inference workflow must accept a supported inspection image.
- **FR-INF-002:** The inference workflow must return a continuous image-level anomaly score.
- **FR-INF-003:** The inference workflow must return a threshold-based decision using the threshold associated with the selected model configuration or artifact.
- **FR-INF-004:** The inference workflow must produce a spatial anomaly map.
- **FR-INF-005:** The inference workflow must distinguish raw scores from normalized visualization values.
- **FR-INF-006:** The inference workflow must not silently substitute missing or incompatible model components.
- **FR-INF-007:** The inference workflow must reject unsupported or unreadable input files with a clear error.
- **FR-INF-008:** Repeated inference using the same model configuration, input, runtime, and deterministic preprocessing must produce equivalent results within the defined numerical tolerance.
- **FR-INF-009:** Patch scores must retain a defined spatial relationship to the generated anomaly map.

### Evaluation

- **FR-EVA-001:** The evaluation workflow must calculate documented image-level metrics.
- **FR-EVA-002:** The evaluation workflow must calculate documented pixel-level metrics using valid supplied masks.
- **FR-EVA-003:** The evaluation workflow must measure relevant runtime, memory, and artifact-size characteristics.
- **FR-EVA-004:** The evaluation workflow must preserve sufficient sample-level information for qualitative error analysis.
- **FR-EVA-005:** Evaluation results must identify the dataset, category, split, experiment, model configuration, and artifact version where applicable.
- **FR-EVA-006:** Comparable variants must use compatible evaluation data and metric definitions.
- **FR-EVA-007:** Final test results must be generated only after model configuration and threshold selection are locked.
- **FR-EVA-008:** The first final evaluation must use the complete official MVTec AD Bottle test partition.
- **FR-EVA-009:** Evaluation may group results by official Bottle test folder but must not use those groups as supervised fitting labels.
- **FR-EVA-010:** Quantitative results must be accompanied by representative qualitative error analysis.

### Model Artifacts

- **FR-ART-001:** A selected model package must contain an ONNX feature-extractor model.
- **FR-ART-002:** A selected model package must contain the fitted anomaly representation, including the feature memory or selected coreset.
- **FR-ART-003:** A selected model package must contain versioned metadata describing preprocessing, backbone, feature layers, embedding dimensions, scoring, threshold, dataset identity, and split identity.
- **FR-ART-004:** A selected model package must contain evaluation results linked to the artifact version.
- **FR-ART-005:** Artifact components intended for .NET must be usable without Python-specific pickle deserialization.
- **FR-ART-006:** The artifact loader must reject missing required files and unsupported schema versions.
- **FR-ART-007:** The artifact loader must reject incompatible tensor dimensions or data types.
- **FR-ART-008:** Artifact integrity must be validated when checksums are required by the artifact schema.
- **FR-ART-009:** Model and dataset attribution required for distribution must accompany released artifacts.
- **FR-ART-010:** Provisional ONNX exports must not be described as complete anomaly-model releases.

### Backend API

The following requirements apply to Stage 2 and have not yet been implemented:

- **FR-API-001:** The backend must expose a versioned endpoint for image-based anomaly inference.
- **FR-API-002:** The backend must validate uploaded file type, size, signature where practical, and decodability before inference.
- **FR-API-003:** A successful response must include the model version, anomaly score, decision, threshold, and relevant timing information.
- **FR-API-004:** A successful response must make localization output available when the artifact supports it.
- **FR-API-005:** The backend must expose health information that distinguishes application availability from model readiness.
- **FR-API-006:** The backend must load and validate a configured artifact package during controlled startup or initialization.
- **FR-API-007:** The backend must return structured validation and inference errors.
- **FR-API-008:** The backend contract must remain independent of a specific web or desktop client.

### Web Application

The following requirements apply to Stage 3 and have not yet been implemented:

- **FR-WEB-001:** The web application must allow a user to select or upload a supported inspection image.
- **FR-WEB-002:** The web application must submit the image to the backend rather than execute independent model logic.
- **FR-WEB-003:** The web application must display the anomaly decision and continuous score.
- **FR-WEB-004:** The web application must display the original image and localization visualization when available.
- **FR-WEB-005:** The web application must display clear loading, validation-error, inference-error, and unavailable-model states.
- **FR-WEB-006:** The web application must identify the model version used for a result.

### Desktop Application

The following requirements apply to Stage 3 and have not yet been implemented:

- **FR-DSK-001:** The desktop application must allow a user to select a supported local inspection image.
- **FR-DSK-002:** The desktop application must submit the image to the backend rather than duplicate the Python anomaly pipeline.
- **FR-DSK-003:** The desktop application must display the anomaly decision and continuous score.
- **FR-DSK-004:** The desktop application must display the original image and localization visualization when available.
- **FR-DSK-005:** The desktop application must display clear loading, validation-error, inference-error, and unavailable-model states.
- **FR-DSK-006:** The desktop application must identify the model version used for a result.

### Observability and Diagnostics

- **FR-OBS-001:** Model-development runs must record enough context to diagnose failed experiments.
- **FR-OBS-002:** Future backend logs must identify startup, artifact-loading, validation, and inference failures without exposing uploaded image content or secrets by default.
- **FR-OBS-003:** Inference timing must distinguish relevant stages such as preprocessing, feature extraction, scoring, and total processing where practical.
- **FR-OBS-004:** Diagnostic output must identify the active model, artifact, and schema versions where applicable.

## Business Rules

- **BR-DAT-001:** Final test labels must not influence model configuration, threshold selection, preprocessing choices, or model selection.
- **BR-DAT-002:** Synthetic validation anomalies must be identified as synthetic and must not be reported as equivalent to real anomalies.
- **BR-MOD-001:** A more complex or optimized model variant must not replace the documented baseline without measured and recorded evidence of its value.
- **BR-MOD-002:** Benchmark metrics must not be presented as proof of production readiness.
- **BR-MOD-003:** A model artifact and its supporting anomaly representation form one compatible package and must be versioned together.
- **BR-MOD-004:** Official defect folder names may support grouped evaluation but must not become supervised training labels in the first normal-only baseline.
- **BR-INF-001:** Visualization normalization must not alter the raw anomaly score used for classification.
- **BR-INF-002:** A threshold must be locked without using final test labels before threshold-dependent final test metrics are calculated.
- **BR-ART-001:** Public artifacts must not be distributed unless applicable model and dataset licenses permit distribution.
- **BR-APP-001:** The system must be described as a demonstration and evaluation project, not as a certified industrial inspection product.

## Non-Functional Requirements

### Reproducibility

- **NFR-APP-001:** Model experiments must record their source revision, configuration, dependency versions, dataset identity, split definition, and random seeds.
- **NFR-APP-002:** Dependency versions must be controlled through versioned dependency files.
- **NFR-APP-003:** A documented local setup must allow another developer to recreate the supported development environment.
- **NFR-APP-004:** File discovery and split membership must remain deterministic.
- **NFR-APP-005:** Released experiment results must reference machine-readable configuration and result data.

### Performance

- **NFR-INF-001:** Initial inference must remain practical on a CPU-only Windows development computer.
- **NFR-INF-002:** Performance claims must identify hardware, runtime, input size, model version, measurement method, warm-up behavior, and sample count.
- **NFR-INF-003:** Optimization decisions must consider accuracy, latency, memory use, and artifact size together.
- **NFR-INF-004:** The complete feature memory must be measured before coreset reduction is introduced as a required optimization.

Exact latency and memory acceptance limits remain open until the complete first baseline has been measured.

### Compatibility

- **NFR-ART-001:** The ONNX model must pass structural validation before acceptance into a candidate artifact package.
- **NFR-ART-002:** PyTorch and ONNX Runtime outputs must remain within documented numerical tolerances for shared reference inputs.
- **NFR-ART-003:** Future .NET preprocessing and inference must be checked against shared Python fixtures.
- **NFR-ART-004:** Artifact schema evolution must be explicit and versioned.

### Maintainability

- **NFR-APP-006:** Dataset handling, preprocessing, feature extraction, anomaly scoring, evaluation, and artifact export must be separated into testable responsibilities.
- **NFR-APP-007:** Client applications must not duplicate model-development logic.
- **NFR-APP-008:** Public contracts and non-obvious model decisions must be documented.
- **NFR-APP-009:** Automated tests must cover critical deterministic behavior and contract validation.

### Security and Privacy

- **NFR-SEC-001:** Secrets, credentials, private service addresses, and developer-specific dataset paths must not be committed to source control.
- **NFR-SEC-002:** Uploaded files must be treated as untrusted input.
- **NFR-SEC-003:** The future backend must enforce configurable upload limits.
- **NFR-SEC-004:** Temporary image data must be removed after processing unless explicit retention behavior is introduced and documented.
- **NFR-SEC-005:** Logs must not include image binary content, credentials, or sensitive local paths by default.
- **NFR-SEC-006:** Public examples must use redistributable or user-created images unless the applicable dataset license explicitly permits the intended use.
- **NFR-SEC-007:** Local dataset archives, extracted files, masks, and private benchmark data must remain outside the public repository.

### Documentation

- **NFR-APP-010:** The project must maintain a README, project specification, architecture overview, development-status document, model-development strategy, and dataset documentation appropriate to the implemented state.
- **NFR-APP-011:** A released model artifact must include a Model Card.
- **NFR-APP-012:** A user guide must be added when an executable workflow requires instructions beyond the README.
- **NFR-APP-013:** Documentation must distinguish verified behavior, intended behavior, open decisions, and deferred work.

## Acceptance Criteria

### Stage 1: Model-Development MVP

- **AC-MOD-001:** MVTec AD Bottle passes the implemented automated dataset validations.
- **AC-MOD-002:** The 167/42 split manifest is loaded and verified without overlap or missing source membership.
- **AC-MOD-003:** The preprocessing pipeline produces the documented 224 × 224 normalized tensor and passes visual crop inspection.
- **AC-MOD-004:** The PatchCore-style baseline builds a reproducible feature memory from the 167 fitting images.
- **AC-MOD-005:** The baseline produces reproducible patch-level and image-level anomaly scores.
- **AC-MOD-006:** A threshold is selected from documented normal-validation evidence without using final test labels.
- **AC-MOD-007:** The unchanged locked baseline is evaluated on the complete official Bottle test partition.
- **AC-MOD-008:** Final evaluation records the selected image-level and pixel-level metrics.
- **AC-MOD-009:** Representative false positives, false negatives, successful detections, and anomaly maps are reviewed.
- **AC-MOD-010:** CPU runtime and complete feature-memory size are measured and documented.
- **AC-MOD-011:** The experiment configuration and results can be reproduced from versioned source, configuration, and the documented external dataset.

### Stage 2: Artifact MVP

- **AC-ART-001:** A candidate package contains the required ONNX model, feature memory, metadata, preprocessing definition, threshold, and evaluation summary.
- **AC-ART-002:** The artifact loader rejects a deliberately incomplete package.
- **AC-ART-003:** The artifact loader rejects incompatible dimensions or an unsupported schema version.
- **AC-ART-004:** PyTorch and Python ONNX Runtime agree within documented tolerances on shared real-image reference cases.
- **AC-ART-005:** The .NET inference implementation agrees within documented tolerances on the same reference cases.
- **AC-ART-006:** Artifact integrity and required attribution are verifiable.

### Stage 2: Backend MVP

- **AC-API-001:** The backend starts with a valid configured artifact and reports model readiness.
- **AC-API-002:** A supported normal reference image returns a successful structured response.
- **AC-API-003:** A supported anomalous reference image returns a successful structured response with localization output.
- **AC-API-004:** Invalid, oversized, unreadable, and unsupported files are rejected with structured errors.
- **AC-API-005:** Automated integration tests cover the primary successful and rejected request paths.

### Stage 3: Client MVPs

- **AC-WEB-001:** The web client can submit an image and display the backend result and localization visualization.
- **AC-WEB-002:** The web client clearly displays backend validation and availability errors.
- **AC-DSK-001:** The desktop client can submit an image and display the backend result and localization visualization.
- **AC-DSK-002:** The desktop client clearly displays backend validation and availability errors.

### Portfolio Release

- **AC-APP-001:** Public documentation accurately describes the implemented state and known limitations.
- **AC-APP-002:** Automated build, test, and quality workflows pass for released repositories.
- **AC-APP-003:** Dependency and vulnerability checks report no unresolved known high-severity issue in the released configuration.
- **AC-APP-004:** Public repositories contain no credentials, private dataset files, restricted artifacts, or developer-specific absolute paths.
- **AC-APP-005:** Released model artifacts include applicable attribution, evaluation results, schema information, and a completed Model Card.

## Open Decisions

The following decisions remain intentionally open:

- final visual approval of the center-crop behavior;
- PatchCore feature-aggregation refinements;
- whether coreset reduction is required;
- coreset algorithm and ratio;
- feature-memory serialization format;
- nearest-neighbor implementation;
- anomaly-map interpolation and smoothing;
- image-score aggregation;
- threshold-selection method;
- exact required MVP metric set;
- numerical parity tolerances;
- exact performance and memory targets;
- artifact schema design;
- API contract details;
- persistence requirements;
- authentication requirements;
- web and desktop presentation design.

The initial dataset, category, split, seed, input size, backbone, pretrained-weight family, feature layers, and first model family are no longer open for the first baseline.

Each remaining decision must be resolved through documented evidence or an architecture decision before it becomes a stable contract.

## Known Limitations

At the time of writing:

- the selected preprocessing has been executed on a real Bottle image, but its center crop has not yet been visually approved;
- the feature extractor and ONNX boundary have been verified, but no anomaly feature memory has been fitted;
- no anomaly scoring implementation exists;
- no threshold has been selected;
- no detection or localization metrics exist;
- no production artifact schema exists;
- no backend, web client, or desktop client has been implemented;
- GPU acceleration is unavailable in the verified local PyTorch environment;
- benchmark data may not represent future real industrial images;
- production, pharmaceutical, safety, and regulatory suitability have not been evaluated.

## Related Documentation

- `DevelopmentStatus.md` records verified implementation progress and immediate next steps.
- `ArchitectureOverview.md` describes the current and intended system structure and component boundaries.
- `ModelDevelopmentStrategy.md` defines the experimental, fitting, threshold, evaluation, and release methodology.
- `DatasetDocumentation.md` records dataset provenance, licensing, structure, validation results, and selection evidence.
- A future `ModelCard.md` will document a selected evaluated artifact.
- A future `UserGuide.md` will describe executable user-facing workflows when they exist.

## Last Updated

This specification reflects the verified project state and selected first-baseline scope as of 2026-08-12.
