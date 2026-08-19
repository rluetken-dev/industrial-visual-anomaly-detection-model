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

The first end-to-end model-development and application-integration cycles are complete. Bottle established the initial pipeline, Capsule tested category generalization and artifact-based inference, and the generalized directory exporter proved that fitting no longer depends on an MVTec-specific manifest.

The current strategy supports two fitting entry points:

1. a versioned manifest for controlled dataset experiments;
2. an external directory containing normal PNG or JPEG images.

Both entry points delegate to the same dataset-independent training implementation and produce category-specific artifacts.

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
- normal-validation threshold selection;
- image-level evaluation and grouped error reporting;
- anomaly heatmaps;
- typed artifact writing, validation, loading, and inference;
- path and stream inference parity;
- persistent FastAPI inference runtime;
- backend, desktop heatmap, and Docker-stack integration;
- 92 passing automated Python tests.

These capabilities establish a functioning reference implementation. They do not establish production readiness, regulatory validation, real-world transfer, or quantitative localization accuracy.

## Core Development Principles

1. Fit only on defect-free images.
2. Keep one artifact scoped to one product category or visually coherent product family.
3. Never mix unrelated categories into one feature memory merely to simplify deployment.
4. Use only normal fitting images to construct the feature memory.
5. Use held-out normal images to select the threshold.
6. Keep final test images and anomaly labels out of fitting and threshold selection.
7. Preserve preprocessing and scoring semantics across fitting, CLI, service, backend, and clients.
8. Record every reproducibility-critical choice.
9. Prefer explicit, testable components over hidden framework behavior.
10. Change one experimental variable at a time whenever practical.
11. Compare optimizations against a complete-memory reference.
12. Treat heatmaps as explanation aids until localization metrics are implemented.
13. Mark results as exploratory when inspected test data influenced development.
14. Reject unsupported or ambiguous input early.
15. Keep generated datasets and artifacts outside Git.

## Evidence-Gated Development Process

```text
problem and category definition
-> source-data qualification
-> normal-image discovery
-> deterministic partitioning
-> preprocessing verification
-> feature-extractor verification
-> feature-memory fitting
-> normal-validation threshold
-> optional test evaluation
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

One artifact represents one category. Bottle, Capsule, and future product categories require separate artifacts unless evidence demonstrates that a shared category definition is coherent.

Category identity is stored in artifact metadata. Runtime model selection must eventually be explicit through configuration, request context, or a controlled routing layer. The current inference service loads one configured artifact at startup.

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
- **Normal validation:** inspect the normal-score distribution and derive the threshold.
- **Normal test:** estimate false-positive behavior on unseen normal data.
- **Anomalous test:** estimate detection behavior and failure modes.
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

The current threshold is:

```text
threshold = maximum(normal validation image scores)
```

Only scores strictly above the threshold are anomalous.

This rule constrains false positives on the observed normal-validation distribution but does not directly optimize anomaly recall. It can be unstable for very small or unrepresentative validation sets. Alternative calibration must be defined without tuning on the final reported test labels.

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

The directory exporter proves fitting independence but does not yet provide a general labeled-test evaluator. The next evaluation component should accept optional normal and anomalous test directories and score a previously exported artifact without refitting it.

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
- aggregation and threshold method;
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

Metadata records schema, dataset, category, backbone, input size, patch grid, embedding dimension, aggregation, threshold, memory fraction, sampling seed, and entry count.

The split sidecar records seed, ratios, counts, and exact relative path membership. The current `torch.save` tensor format is trusted deployment input. Framework-neutral storage remains optional future portability work.

The refactored MVTec exporter reproduced the established Capsule feature memory byte-for-byte, with SHA-256:

```text
51DE3F2B4FEF804E9E95900597E738E86F7044A669D2739956CBA0CC6DE65478
```

## Offline and Service Inference Strategy

Path-based CLI inference and binary-stream inference share the same artifact, preprocessing, feature extractor, scoring, aggregation, and threshold logic.

The internal FastAPI service loads one configured artifact and extractor at startup and reuses them. Responses contain model ID, category, score, threshold, anomaly decision, and a threshold-normalized Base64 PNG heatmap.

The service remains internal. ASP.NET Core owns public upload validation, limits, Problem Details, trace identifiers, readiness, and stable client-neutral contracts. The WPF client calls only the backend.

## Local Runtime and Deployment Strategy

CPU-only inference remains the compatibility baseline. Exact search against complete memory is practical for single images but expensive for category-wide evaluation.

The separate Docker Compose stack builds version-pinned inference and backend images, mounts a selected artifact read-only, supplies service networking and health checks, and verifies the local end-to-end request. The WPF desktop client remains a native Windows application outside Docker.

Production authentication, network hardening, signed artifacts, monitoring, and scaled inference remain future work.

## Model Approval Gate

A category artifact may become an internal reference only when:

- source data and category scope are documented;
- normal images are reviewed and readable;
- fitting and validation roles are reproducible;
- preprocessing and feature extraction are verified;
- memory and aggregation configuration are explicit;
- threshold derivation is documented;
- artifact export and loading succeed;
- known normal and anomalous examples behave plausibly;
- evaluation evidence and limitations are disclosed;
- automated quality checks pass;
- inference behavior matches the stored metadata.

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

Current local Capsule and Bottle artifacts do not yet satisfy this release gate.

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

- automatic multi-category artifact selection;
- all-category MVTec fitting;
- MVTec LOCO AD fitting;
- MVTec AD 2 private evaluation;
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
17. Ninety-two passing Python tests.

## Immediate Next Steps

1. Evaluate the generalized workflow on a genuinely non-MVTec normal-image collection.
2. Define the supported external dataset contract and recommended image counts.
3. Add general artifact evaluation against optional normal and anomalous test directories.
4. Design explicit multi-artifact selection across service, backend, and clients.
5. Add stronger artifact provenance, preprocessing metadata, and checksums.
6. Investigate coverage-preserving memory reduction and faster search.
7. Add fixed-fixture inference regression coverage where practical.
8. Prepare a Model Card before considering public artifact distribution.

## Related Documentation

- `ArchitectureOverview.md` describes system boundaries and runtime flow.
- `DatasetDocumentation.md` records dataset sources, licenses, and validation.
- `DevelopmentStatus.md` records verified implementation progress.
- `ProjectSpecification.md` defines scope and requirements.
- a future `ModelCard.md` will document a released evaluated artifact.

## Last Updated

2026-08-19
