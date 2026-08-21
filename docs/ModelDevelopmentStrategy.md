# Industrial Visual Anomaly Detection – Model Development Strategy

## Purpose

This document defines the strategy for developing, fitting, validating, evaluating, comparing, exporting, serving, and evolving anomaly-detection models for the Industrial Visual Anomaly Detection project.

The strategy is designed to prevent:

- test-set leakage;
- accidental mixing of unrelated product categories;
- irreproducible data partitions;
- silent preprocessing or scoring changes;
- misleading benchmark claims;
- incompatible model artifacts;
- optimization without measurable evidence.

## Strategy Status

The first end-to-end model-development and application-integration cycles are complete. Bottle established the initial pipeline, Capsule tested category generalization and artifact-based inference, and the generalized directory exporter removed the dependency on MVTec-specific fitting manifests. VisA Candle exercised generalized fitting, dataset-independent evaluation, and exploratory threshold calibration. VisA Cashew provided an additional artifact for registry-based multi-model integration.

The strategy supports two fitting entry points:

1. a versioned manifest for controlled dataset experiments;
2. an external directory containing normal PNG or JPEG images.

Both entry points delegate to the same dataset-independent training implementation and produce category-specific artifacts. Labeled evaluation is supplied separately through dataset-independent CSV manifests.

Deployment supports two service configurations:

1. legacy startup with one artifact;
2. registry startup with multiple enabled category artifacts and one default model.

Multi-model deployment does not change the fitting rule: every artifact is still trained, calibrated, evaluated, and approved independently for one declared category.

## Implemented and Verified Capabilities

- validation of MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- recursive discovery of external PNG and JPEG images;
- deterministic automatic fitting and validation splits;
- portable split records using relative paths;
- deterministic preprocessing at configurable square sizes;
- frozen pretrained ResNet18 feature extraction;
- multi-scale `layer2` and `layer3` patch embeddings;
- complete and optionally sampled feature memories;
- exact chunked nearest-neighbor scoring;
- top-fraction image-score aggregation;
- configurable normal-validation quantile threshold selection;
- dataset-independent labeled-image CSV manifests;
- artifact evaluation with score distributions, confusion matrices, classification metrics, group rates, and error lists;
- anomaly heatmaps;
- typed artifact writing, validation, loading, and inference;
- path and stream inference parity;
- persistent legacy single-artifact FastAPI runtime;
- validated model-registry configuration;
- startup loading of multiple enabled category artifacts;
- default and explicit runtime selection by stable model identifier;
- FastAPI model-catalog and model-specific prediction endpoints;
- backend, desktop, heatmap, and Docker-stack multi-model integration;
- 144 automated Python test methods.

These capabilities establish a functioning reference implementation. They do not establish production readiness, regulatory validation, real-world transfer, or quantitative localization accuracy.

## Core Development Principles

1. Fit only on defect-free images.
2. Keep one artifact scoped to one product category or visually coherent product family.
3. Never mix unrelated categories into one feature memory merely to simplify deployment.
4. Use only normal fitting images to construct the feature memory.
5. Use held-out normal images to calculate the initial threshold.
6. Keep final test images and anomaly labels out of fitting and initial threshold calculation.
7. Separate threshold calibration evidence from independent final-test evidence.
8. Preserve preprocessing and scoring semantics across fitting, CLI, service, backend, and clients.
9. Record every reproducibility-critical choice.
10. Prefer explicit, testable components over hidden framework behavior.
11. Change one experimental variable at a time whenever practical.
12. Compare optimizations against a complete-memory reference.
13. Treat heatmaps as explanation aids until localization metrics are implemented.
14. Mark results as exploratory when inspected test data influenced development.
15. Reject unsupported or ambiguous input early.
16. Keep generated datasets and artifacts outside Git.

## Evidence-Gated Development Process

```text
problem and category definition
-> source-data qualification
-> normal-image discovery
-> deterministic partitioning
-> preprocessing verification
-> feature-extractor verification
-> feature-memory fitting
-> initial normal-validation quantile threshold
-> optional calibration evaluation
-> freeze configuration
-> independent final evaluation
-> qualitative heatmap review
-> artifact export
-> offline inference verification
-> service verification
-> backend and client verification
-> release decision
```

Each stage should produce evidence before the next stage becomes authoritative.

## External Dataset Contract

### Minimum Input

The generalized exporter requires one directory containing normal images. It searches recursively for:

- `.png`;
- `.jpg`;
- `.jpeg`.

Suffix matching is case-insensitive. Unsupported files are ignored. The source directory must exist and contain at least two unique supported images so both fitting and validation remain non-empty.

Example:

```text
normal-images/
├── product-001.png
├── product-002.png
└── optional-subdirectory/
    └── product-003.jpg
```

The directory may come from MVTec, a camera export, a manually curated dataset, or another source. The general model code must not infer dataset-specific semantics from folder names.

### Recommended Input Quality

Normal fitting data should:

- represent the intended camera, optics, lighting, background, pose, and product variation;
- exclude known defects and unreadable files;
- avoid duplicates or near-duplicate bursts dominating the memory;
- cover acceptable manufacturing variation;
- be reviewed for accidental anomalies before fitting;
- be legally usable for the intended purpose.

The implementation technically accepts two images, but that is only a structural minimum. A meaningful minimum image count remains category-dependent and must be established through evaluation.

## Category Strategy

One artifact represents one category. Bottle, Capsule, Candle, Cashew, and future product categories require separate artifacts unless evidence demonstrates that a shared category definition is coherent.

Category identity is stored in artifact metadata. Deployment identity is represented by a stable model ID in `models.json`. The registry groups independently trained artifacts for serving but does not merge their fitting data, thresholds, or feature memories.

The registry defines:

- which artifact entries are enabled;
- the relative directory of each artifact;
- a human-readable display name;
- one default model identifier.

Clients select an available model explicitly through the backend. If a compatible request omits the model identifier, the configured registry default is used. Automatic visual recognition of the appropriate category is not implemented and must not be inferred from the image.

## Data Partition Strategy

### Automatic External-Directory Split

The generalized workflow:

1. discovers supported images in deterministic sorted order;
2. shuffles them with an explicit split seed;
3. assigns the requested fraction to normal validation;
4. keeps fitting and validation non-empty;
5. sorts both output partitions;
6. verifies complete coverage and no overlap;
7. records exact membership in `training_split.json`.

The current defaults are:

| Parameter | Default |
| --- | ---: |
| Validation fraction | 0.2 |
| Split seed | 42 |

`training_split.json` stores relative paths rather than machine-specific absolute paths.

### Versioned Manifest Split

Controlled MVTec experiments may continue to use committed manifests:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
configs/splits/mvtec-ad-capsule-seed-42.json
```

Manifest loading validates counts, duplicates, overlap, relative path safety, and dataset-root resolution. Both entry points ultimately pass explicit fitting and validation paths to the same training function.

### Partition Roles

- **Fitting:** construct the normal feature memory.
- **Normal validation:** inspect the normal-score distribution and calculate the initial quantile threshold.
- **Calibration evaluation:** compare a limited, predeclared set of operating choices when necessary; results used here are no longer an untouched final estimate.
- **Normal final test:** estimate false-positive behavior on unseen normal data after configuration is frozen.
- **Anomalous final test:** estimate detection behavior and failure modes after configuration is frozen.
- **Masks:** evaluate localization only; never influence fitting or image-level threshold selection.

## Preprocessing Strategy

The selected reference preprocessing is:

```text
image decode
-> RGB conversion
-> direct resize to configured square input
-> tensor conversion
-> ImageNet normalization
```

Direct resizing was selected because center cropping removed relevant Bottle boundaries. Current verified resolutions are 224 x 224 and 320 x 320.

Before fitting a substantially non-square category, compare direct resizing against padding or another explicit aspect-ratio policy. Do not silently change preprocessing for an existing artifact lineage.

Future artifact schemas should explicitly record interpolation, antialiasing, color conversion, normalization mean and standard deviation, and pretrained weight identity.

## Feature Extraction Strategy

The reference extractor is a pretrained ResNet18 with gradients disabled and evaluation mode enabled.

`layer2` and `layer3` features are fused:

1. capture both intermediate feature maps;
2. resize `layer3` spatially to the `layer2` grid;
3. concatenate channels;
4. flatten spatial positions into 384-dimensional patch embeddings.

| Input | Patch grid | Embeddings per image |
| --- | --- | ---: |
| 224 x 224 | 28 x 28 | 784 |
| 320 x 320 | 40 x 40 | 1,600 |

Fine-tuning is deferred. A backbone change requires a new artifact lineage and renewed evaluation.

## Feature Memory Strategy

The feature memory concatenates embeddings from all normal fitting images in deterministic loader order.

The complete feature memory remains the reference because deterministic random sampling reduced Capsule recall materially:

| Memory | Entries | Recall | F1 | FN |
| --- | ---: | ---: | ---: | ---: |
| 100% | 280,000 | 0.9541 | 0.9674 | 5 |
| 75% | 210,000 | 0.8991 | 0.9423 | 11 |
| 50% | 140,000 | 0.8624 | 0.9216 | 15 |
| 25% | 70,000 | 0.6330 | 0.7753 | 40 |

Future reduction should use a coverage-preserving coreset or another method evaluated against complete memory.

## Nearest-Neighbor and Scoring Strategy

Each query patch is compared with the feature memory using exact Euclidean distance. Memory entries are processed in configurable chunks to bound temporary allocation.

Supported image-level aggregation includes maximum patch score and top-fraction mean. The selected reference rule is:

```text
image score = mean(highest-scoring 1% of patches)
```

Aggregation parameters belong to artifact metadata and must not change during inference.

## Threshold Strategy

The implemented threshold rule is:

```text
threshold = quantile(normal validation image scores, configured quantile)
```

Only scores strictly above the threshold are anomalous. A quantile of `1.0` preserves the former maximum-normal rule.

Lower quantiles generally increase recall while also increasing false-positive risk. The selected quantile is category-specific calibration state, not a universal project constant. It is recorded together with the threshold method in schema-version-2 artifact metadata.

A quantile may be selected from declared operational requirements, separate calibration evidence, or a predefined project policy. If labeled test results are inspected while choosing it, those results become exploratory calibration evidence and must not be reused as an independent final benchmark.

## Evaluation Strategy

### Image-Level Evaluation

For labeled test data, report:

- true positives, true negatives, false positives, and false negatives;
- accuracy, precision, recall, and F1;
- score distributions by group;
- per-defect detection rates;
- inference timings;
- concrete false-positive and false-negative examples.

### Generalized Artifact Evaluation

The generic evaluator loads a previously exported artifact and a CSV manifest containing `image`, `group`, and `is_anomalous` columns. Paths are resolved against an explicit dataset root. The loader rejects missing columns, invalid labels, unsupported suffixes, duplicate paths, missing files, absolute paths, and traversal outside the root.

Evaluation reports score distributions, confusion-matrix counts, accuracy, precision, recall, specificity, F1 score, group-level anomaly rates, false positives, false negatives, and timings. Evaluation never refits or modifies the artifact.

### Pixel-Level Evaluation

Future localization evaluation should include pixel AUROC, average precision, thresholded segmentation metrics, and PRO-style overlap where appropriate.

### Qualitative Review

Review representative normal images, true anomalies, false positives, false negatives, subtle defects, boundaries, and heatmaps. Do not select only visually convincing examples.

## Verified Reference Evidence

### Exploratory Bottle Baseline at 224 x 224

| Aggregation | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum | 0.9398 | 1.0000 | 0.9206 | 0.9587 | 0 | 5 |
| Top 1% mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Top 5% mean | 0.9759 | 0.9692 | 1.0000 | 0.9844 | 2 | 0 |

### Exploratory Capsule Reference at 320 x 320

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9470 |
| Precision | 0.9811 |
| Recall | 0.9541 |
| F1 | 0.9674 |
| False positives | 2 |
| False negatives | 5 |

### Generalized Bottle Export at 320 x 320

| Property | Result |
| --- | ---: |
| Source normal images | 209 |
| Fitting images | 167 |
| Validation images | 42 |
| Feature-memory entries | 267,200 |
| Feature-memory size | 391.41 MiB |
| Threshold | 3.2163138389587402 |

Smoke predictions classified Bottle `test/good/000.png` as normal and `test/broken_large/000.png` as anomalous. This verifies artifact compatibility, not full generalized evaluation.

The exploratory MVTec results are development evidence rather than untouched benchmark claims.

### Exploratory VisA Candle Calibration at 320 x 320

The official VisA Candle one-class split supplied 900 normal training images, 100 normal test images, and 100 anomalous test images. The generalized exporter created a deterministic 720/180 fitting and validation split and sampled 25 percent of the complete fitting feature memory.

The q100, q99, and q95 artifacts have identical feature-memory SHA-256 hashes. Only their normal-validation thresholds differ.

| Variant | Quantile | Threshold | TP | TN | FP | FN | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q100 | 1.00 | 3.373678 | 31 | 100 | 0 | 69 | 1.0000 | 0.3100 | 1.0000 | 0.4733 |
| q99 | 0.99 | 3.001366 | 56 | 100 | 0 | 44 | 1.0000 | 0.5600 | 1.0000 | 0.7179 |
| q95 | 0.95 | 2.763051 | 69 | 95 | 5 | 31 | 0.9324 | 0.6900 | 0.9500 | 0.7931 |

q95 is the provisional candidate for a review-oriented workflow. It is not independently validated because the official Candle test results were inspected while comparing quantiles. The strategy therefore freezes q95 for the next validation step and prohibits further tuning against the same Candle test split.

## Multi-Model Deployment Verification

The local registry-based deployment has been verified with these artifact identities:

```text
mvtec-ad-capsule-320
mvtec-ad-bottle-generalized-320
visa-candle-generalized-q95-320
visa-cashew-generalized-q95-320
```

The inference service loaded all enabled artifacts during startup and exposed them through through `GET /api/v1/models`. Capsule, Bottle, Candle, and Cashew were selected through the native desktop workflow. Capsule and Cashew were additionally selected explicitly through the containerized backend workflow.

These checks verify registry loading, model routing, response identity, and heatmap generation. They do not replace category-specific model-quality evaluation. In particular, the successful Cashew requests are integration evidence rather than an independent Cashew benchmark.

## Experiment Management

Every meaningful experiment should record:

- dataset identity and legal provenance;
- category;
- source image location or dataset version;
- fitting, validation, and test roles;
- split seed and validation fraction;
- exact partition membership;
- preprocessing and input size;
- backbone and pretrained weights;
- feature layers and embedding dimension;
- memory fraction and sampling seed;
- distance and chunk configuration;
- aggregation, threshold method, and threshold quantile;
- whether each labeled partition was used for calibration or independent final evaluation;
- metrics, timings, and failure examples;
- output artifact location and checksum;
- relevant source revision and dependency versions.

Machine-specific absolute paths must not enter released artifact provenance.

## Reproducibility Strategy

Reproducibility depends on:

- deterministic image discovery;
- explicit split and sampling seeds;
- recorded partition membership;
- pinned dependencies;
- versioned source and schemas;
- deterministic preprocessing and loader order;
- isolated fitting, validation, and test roles;
- artifact checksums;
- clean-environment reconstruction.

The generalized exporter now records partition membership. Complete experiment reports, embedded software provenance, and artifact checksums remain future improvements.

## Artifact Strategy

The core artifact contains:

```text
model-artifact/
  metadata.json
  feature_memory.pt
```

The generalized exporter additionally creates:

```text
  training_split.json
```

Schema-version-2 metadata records schema, dataset, category, backbone, input size, patch grid, embedding dimension, aggregation, threshold, threshold method, threshold quantile, memory fraction, sampling seed, and entry count. The loader preserves schema-version-1 compatibility by applying maximum-normal threshold defaults.

The split sidecar records seed, ratios, counts, and exact relative path membership. The current `torch.save` tensor format is trusted deployment input. Framework-neutral storage remains optional future portability work.

The refactored MVTec exporter reproduced the established Capsule feature memory byte-for-byte, with SHA-256:

```text
51DE3F2B4FEF804E9E95900597E738E86F7044A669D2739956CBA0CC6DE65478
```

## Offline and Service Inference Strategy

Path-based CLI inference and binary-stream inference share the same artifact, preprocessing, feature extractor, scoring, aggregation, and threshold logic.

The internal FastAPI service supports two mutually exclusive startup modes:

- `IVAD_MODEL_ARTIFACT` loads one legacy artifact;
- `IVAD_MODEL_REGISTRY` validates a registry and loads one runtime for every enabled artifact.

Exactly one source must be configured. Loaded runtimes are reused across requests.

The service exposes:

```text
GET  /health/live
GET  /api/v1/models
POST /api/v1/predictions
```

The catalog endpoint returns the default and available models. The prediction endpoint accepts multipart `image` and optional `modelId`. Registry mode selects the requested runtime or falls back to the default. Responses contain the actual model ID, category, score, threshold, anomaly decision, and a threshold-normalized Base64 PNG heatmap.

The service remains internal. ASP.NET Core owns the public catalog, upload validation, limits, Problem Details, trace identifiers, readiness, and stable client-neutral contracts. The WPF client retrieves models and submits analyses only through the backend.

## Local Runtime and Deployment Strategy

CPU-only inference remains the compatibility baseline. Exact search against complete feature memories is practical for individual images but expensive for category-wide evaluation.

Registry mode loads every enabled artifact and extractor during startup. This provides predictable request behavior but increases startup duration and resident memory with each enabled model. Multiple process workers would duplicate all loaded feature memories. Lazy loading, unloading, and registry hot reload require separate design and verification before adoption.

The Docker Compose stack builds explicit inference and backend source revisions, mounts the registry and artifact root read-only, supplies service networking and health checks, and verifies model-specific requests. Changing between already loaded models does not require editing `.env` or recreating containers.

The WPF desktop client remains a native Windows application outside Docker. Production authentication, network hardening, signed artifacts, monitoring, GPU execution, and scaled inference remain future work.

## Model Approval Gate

A category artifact may become an internal reference only when:

- source data and category scope are documented;
- normal images are reviewed and readable;
- fitting and validation roles are reproducible;
- preprocessing and feature extraction are verified;
- memory and aggregation configuration are explicit;
- threshold derivation and quantile are documented;
- calibration evidence is separated from independent final-test evidence;
- artifact export and loading succeed;
- known normal and anomalous examples behave plausibly;
- evaluation evidence and limitations are disclosed;
- automated quality checks pass;
- inference behavior matches the stored metadata.

## Deployment Registry Approval Gate

A model registry may become an internal deployment reference only when:

- every enabled model has a stable unique identifier;
- the default identifier references an enabled model;
- artifact directories are relative, unique, and free from parent traversal;
- every enabled artifact exists and passes artifact validation;
- registry and artifact metadata agree on runtime model identity and category;
- each enabled model has category-specific approval evidence;
- the catalog exposes the intended display name, category, input size, and default state;
- explicit and default model selection return the expected runtime;
- unknown model identifiers fail clearly;
- registry and artifact files remain external, ignored, and read-only in container deployments;
- the combined startup time and memory requirement are acceptable for the target environment.

Successful registry loading does not approve the model quality of every entry automatically. Each artifact retains its own fitting, calibration, evaluation, and release evidence.

## Public Release Gate

A model artifact may be released publicly only when:

- dataset redistribution rights are confirmed;
- source-code licensing is explicit;
- artifact and provenance schemas are versioned;
- checksums are published;
- evaluation results link to the exact artifact;
- preprocessing and dependency compatibility are documented;
- limitations and intended use are documented in a Model Card;
- no restricted dataset content is embedded unintentionally.

Current local Capsule, Bottle, VisA Candle, and VisA Cashew artifacts do not yet satisfy this public artifact release gate. Publishing the model-service source release does not publish or redistribute these local artifacts.

## Failure Analysis Strategy

Analyze failures by separating:

- source-data contamination or insufficient normal coverage;
- preprocessing or image-decoding errors;
- pose, lighting, background, scale, or product drift;
- feature-memory coverage gaps;
- threshold calibration errors;
- small or localized defects;
- boundary and texture effects;
- artifact incompatibility;
- service configuration, timeout, transport, or contract failures.

False negatives are the primary inspection risk, but false positives matter operationally and must also be reported.

## Deferred Model Work

- automatic visual category recognition;
- dynamic registry hot reload;
- lazy model loading and unloading;
- all-category MVTec fitting;
- MVTec LOCO AD fitting;
- MVTec AD 2 private evaluation;
- independent Cashew benchmark evaluation;
- non-square preprocessing policy;
- principled coreset selection;
- approximate nearest-neighbor search;
- pixel-level benchmark metrics;
- backbone fine-tuning;
- continual learning and automated retraining;
- GPU optimization;
- real industrial product validation;
- production monitoring and drift handling;
- regulatory validation.

## Completed Model-Development Milestones

1. Dataset acquisition, checksums, extraction, and validation.
2. Deterministic Bottle and Capsule manifests.
3. Explicit preprocessing, feature extraction, embeddings, memory, scoring, and evaluation.
4. Bottle and Capsule exploratory evaluation.
5. Feature-memory sampling experiment.
6. Heatmap generation and visualization.
7. Typed artifact metadata, writer, loader, and inference.
8. Persistent FastAPI runtime and heatmap response.
9. ASP.NET Core, WPF desktop, and Docker-stack integration.
10. General external PNG/JPEG discovery.
11. Deterministic automatic fitting and validation split.
12. Portable `training_split.json` generation.
13. Dataset-independent training orchestration.
14. Generalized 320 x 320 Bottle artifact export.
15. Normal and anomalous Bottle smoke predictions.
16. Byte-for-byte Capsule exporter compatibility verification.
17. Dataset-independent labeled-manifest artifact evaluation.
18. Configurable normal-score quantile threshold selection.
19. Schema-version-2 threshold metadata and schema-version-1 loading compatibility.
20. Exploratory VisA Candle threshold calibration with q95 selected provisionally.
21. Validated model-registry configuration and path-safety rules.
22. Startup loading of multiple enabled category artifacts.
23. Default and explicit runtime selection by stable model ID.
24. FastAPI model-catalog endpoint and optional prediction `modelId`.
25. Backend, desktop, and Docker-stack multi-model integration.
26. Capsule, Bottle, Candle, and Cashew runtime selection verification.
27. One hundred forty-four automated Python test methods.

## Immediate Next Steps

1. Complete the registry-capable documentation update.
2. Run the complete Python quality checks and review the intended repository diff.
3. Publish an immutable registry-capable model-service release.
4. Update downstream stack builds to use that release tag.
5. Keep q95 fixed and validate the threshold strategy on previously unused evidence.
6. Perform an independently controlled Cashew evaluation.
7. Define recommended external dataset image counts and a strict final-test protocol.
8. Add stronger artifact provenance, preprocessing metadata, and checksums.
9. Investigate coverage-preserving memory reduction and faster search.
10. Prepare a Model Card before considering public artifact distribution.

## Related Documentation

- `ArchitectureOverview.md` describes system boundaries and runtime flow.
- `DatasetDocumentation.md` records dataset sources, licenses, and validation.
- `DevelopmentStatus.md` records verified implementation progress.
- `experiments/visa-candle-threshold-calibration.md` records the exploratory calibration evidence and its methodological limitation.
- `ProjectSpecification.md` defines scope and requirements.
- a future `ModelCard.md` will document a released evaluated artifact.

## Last Updated

2026-08-21