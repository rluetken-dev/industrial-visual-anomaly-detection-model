# Industrial Visual Anomaly Detection – Model Development Strategy

## Purpose

This document defines the strategy for developing, fitting, validating, evaluating, comparing, exporting, and evolving anomaly-detection models for the Industrial Visual Anomaly Detection project.

It is intended to prevent:

- test-set leakage;
- irreproducible experiments;
- misleading benchmark claims;
- undocumented model changes;
- incompatible model artifacts;
- premature optimization;
- confusion between benchmark performance and production readiness.

The first end-to-end model-development cycle has been completed. Bottle established the initial pipeline, and Capsule was used to test generalization, input-resolution changes, feature-memory sampling, artifact export, and reusable single-image inference.

## Strategy Status

This document distinguishes between:

- **Implemented and verified** – demonstrated through executable code, automated tests, or recorded experiments;
- **Selected reference strategy** – used by the current reference artifact;
- **Exploratory result** – useful evidence that was obtained during development but not through an untouched blind benchmark;
- **Open decision** – requires further implementation or evaluation;
- **Deferred work** – excluded from the current development cycle.

## Implemented and Verified Capabilities

- CPU-based PyTorch and TorchVision execution;
- pretrained frozen ResNet18 feature extraction;
- `layer2` and `layer3` feature fusion;
- configurable square preprocessing at 224 × 224 and 320 × 320;
- deterministic category-specific split manifests;
- local patch embeddings with dimension 384;
- complete normal feature-memory construction;
- exact chunked nearest-neighbor distances;
- maximum and top-fraction patch-score aggregation;
- normal-validation threshold selection;
- grouped image-level evaluation;
- anomaly heatmap generation;
- deterministic random feature-memory sampling;
- typed model-artifact metadata;
- artifact writing and loading;
- individual image inference and prediction CLI;
- provisional ONNX export and PyTorch/ONNX numerical parity;
- 54 passing automated tests.

These capabilities establish a functioning reference implementation. They do not establish production readiness, regulatory validation, real-world transfer, or complete Python/.NET parity.

## Reference Model Configuration

The current reference configuration is:

| Setting | Selected value |
| --- | --- |
| Dataset | MVTec AD |
| Category | `capsule` |
| Learning mode | Normal-only unsupervised anomaly detection |
| Fitting images | 175 |
| Normal validation images | 44 |
| Test images | 132 |
| Random seed | 42 |
| Backbone | Frozen ResNet18 with default TorchVision weights |
| Feature layers | `layer2` and `layer3` |
| Input size | 320 × 320 |
| Preprocessing | RGB, direct bilinear resize with antialiasing, tensor conversion, ImageNet normalization |
| Embedding dimension | 384 |
| Patch grid | 40 × 40 |
| Feature-memory entries | 280,000 |
| Feature-memory fraction | 1.0 |
| Distance search | Exact chunked nearest neighbor |
| Image aggregation | Mean of highest 1% of patch scores |
| Threshold | Maximum normal-validation score |
| Stored threshold | 2.501821517944336 |

The split manifest is:

```text
configs/splits/mvtec-ad-capsule-seed-42.json
```

The selected model detects deviation from normal appearance and returns a spatial patch-score map. It does not classify the exact defect type.

## Model Development Principles

1. Verify dataset source, license, structure, and integrity before use.
2. Use only normal fitting images to construct the normal reference memory.
3. Keep threshold selection independent from official test labels.
4. Preserve preprocessing and scoring semantics across fitting and inference.
5. Begin with explicit, testable mechanics before advanced optimization.
6. Record every meaningful model change as a reproducible experiment.
7. Separate image-level detection from pixel-level localization claims.
8. Review false positives and false negatives in addition to aggregate metrics.
9. Treat runtime and memory consumption as model-quality constraints.
10. Keep artifacts inseparable from their preprocessing, backbone, dimensions, aggregation, and threshold metadata.
11. Mark results as exploratory when test data influenced later development decisions.
12. Never present benchmark results as evidence of production suitability.

## Evidence-Gated Development Process

```text
dataset qualification                         completed
        ↓
data validation and deterministic manifests   completed
        ↓
preprocessing verification                     completed
        ↓
reusable feature extraction                    completed
        ↓
complete feature memory                        completed
        ↓
exact anomaly scoring                          completed
        ↓
normal-validation threshold                    completed
        ↓
Bottle baseline evaluation                     completed
        ↓
Capsule generalization and resolution study    completed
        ↓
memory-sampling tradeoff study                  completed
        ↓
local artifact export and Python inference     completed
        ↓
pixel-level evaluation                         pending
        ↓
framework-neutral artifact contract            pending
        ↓
Python/ONNX/.NET parity                         pending
```

## Dataset Qualification Strategy

MVTec AD, MVTec LOCO AD, and MVTec AD 2 have been downloaded, checksummed, extracted, inspected, and validated. Qualification records source, attribution, license, archive checksum, structure, inventories, image properties, masks, and suitability for normal-only fitting.

MVTec AD remains the active development dataset. MVTec LOCO AD and MVTec AD 2 are qualified for later evaluation but are not part of the current fitted reference artifact.

Datasets and derived reports are stored outside Git. Public code must not depend on machine-specific absolute paths.

## Data Partition Strategy

Versioned manifests define fitting and normal-validation membership using relative paths and seed `42`.

| Category | Fitting | Normal validation | Official test |
| --- | ---: | ---: | ---: |
| Bottle | 167 | 42 | 83 |
| Capsule | 175 | 44 | 132 |

Partition roles are:

- **Fitting:** build the normal feature memory;
- **Normal validation:** analyze normal scores and choose the threshold;
- **Official test:** report detection and grouped error behavior;
- **Ground truth masks:** evaluate localization when pixel-level metrics are implemented.

Test labels must not be used to construct memory or determine the threshold. Because Bottle and Capsule test results were inspected and then informed later design work, the current comparisons are explicitly exploratory rather than untouched final benchmarks.

## Preprocessing Strategy

The implemented deterministic preprocessing is:

```text
decode image
→ convert to RGB
→ direct resize to configured square input
→ convert to torch.float32 tensor
→ normalize with ImageNet mean and standard deviation
```

Bottle uses 224 × 224. Capsule uses 320 × 320 because higher resolution materially improved sensitivity to small defects.

The default TorchVision center-crop path was rejected for Bottle because it removed part of the object boundary. Direct resizing introduces no aspect-ratio distortion for the inspected square Bottle and Capsule images.

Non-square categories require a separate documented choice between direct resizing, aspect-ratio preservation, and padding.

No random augmentation belongs to the current reference model. Any future augmentation experiment must preserve the meaning of normal data, use explicit seeds, and be compared with unchanged deterministic evaluation preprocessing.

## Feature Extraction Strategy

The frozen ResNet18 backbone supplies `layer2` and `layer3` features. `layer3` is bilinearly resized to the spatial resolution of `layer2`, concatenated along the channel dimension, and rearranged into 384-dimensional local patch embeddings.

At 224 × 224:

```text
layer2:          (1, 128, 28, 28)
layer3:          (1, 256, 14, 14)
patch embeddings:      (784, 384)
```

At 320 × 320:

```text
layer2:          (1, 128, 40, 40)
layer3:          (1, 256, 20, 20)
patch embeddings:     (1600, 384)
```

Backbone replacement or fine-tuning is deferred until data, preprocessing, scoring, threshold behavior, and runtime have been examined first.

## Feature Memory Strategy

The feature memory concatenates embeddings from all normal fitting images in deterministic loader order.

It is bound to:

- dataset and category;
- split manifest;
- input and preprocessing configuration;
- backbone and pretrained weights;
- selected feature layers;
- embedding construction;
- embedding dimension;
- sampling or coreset configuration;
- framework and artifact versions.

Incompatible feature memories must never be combined.

The Capsule reference memory contains:

```text
175 × 1600 = 280000 embeddings
shape: (280000, 384)
dtype: float32
size: approximately 410.16 MiB
```

The current local serialization uses `torch.save`. A framework-neutral format is required before .NET consumption.

## Memory-Reduction Strategy

The complete memory is always established first as the quality reference. Optimization is evaluated against that fixed configuration.

Deterministic random sampling with seed `42` produced the following Capsule results using top-one-percent aggregation:

| Fraction | Entries | Validation + test scoring | Recall | F1 | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 280,000 | 225.06 s | 0.9541 | 0.9674 | 5 |
| 75% | 210,000 | 184.33 s | 0.8991 | 0.9423 | 11 |
| 50% | 140,000 | 120.36 s | 0.8624 | 0.9216 | 15 |
| 25% | 70,000 | 64.15 s | 0.6330 | 0.7753 | 40 |

Random sampling reduces runtime but loses too much normal feature coverage. The reference artifact therefore retains 100% of the memory. Future reduction should use a coverage-preserving coreset or another method evaluated against the complete-memory reference.

## Nearest-Neighbor Strategy

The current Python implementation uses exact Euclidean nearest-neighbor distance. Reference memory is processed in configurable chunks to control temporary memory without changing results.

The recorded scoring configuration must include:

- distance definition;
- query and memory chunking;
- tensor data type;
- patch-grid dimensions;
- memory size;
- runtime environment.

Approximate search may be considered if it preserves a documented quality threshold. A future .NET numerical implementation must be checked against fixed Python fixtures and tolerances.

## Image-Level Scoring Strategy

Supported aggregation methods are:

- maximum patch distance;
- mean of the highest configurable fraction of patch distances.

Top-one-percent mean is the selected reference because it was more robust than a single extreme patch and performed strongly in the explored Bottle and Capsule evaluations.

Aggregation parameters belong to artifact metadata and must not be silently changed at inference time.

## Threshold Strategy

The reference threshold is the maximum image score among normal validation images:

```text
threshold = max(normal validation scores)
```

Only scores strictly greater than the threshold are classified as anomalous.

This strategy constrains false positives on the available validation-normal distribution but cannot directly optimize anomaly recall. Alternative thresholds may be evaluated later, but they must be selected without using the final test labels of the experiment being reported.

The exported Capsule artifact stores threshold `2.501821517944336`.

## Anomaly-Map Strategy

Nearest-neighbor distances are reconstructed as a two-dimensional patch grid. Visualization can:

- resize a patch grid with bilinear interpolation;
- normalize each map independently;
- normalize against a fixed decision threshold;
- colorize the map;
- blend it with the source image.

Fixed threshold-based normalization is preferred for comparison across images. Heatmap colors remain presentation data and must not replace raw scores.

Quantitative localization metrics have not yet been implemented. Any smoothing, normalization, or pixel threshold that affects metrics must be part of the recorded experiment configuration.

## Evaluation Strategy

### Image-Level Evaluation

The implemented evaluation reports:

- normal score distributions;
- per-defect-group score distributions;
- true positives, true negatives, false positives, and false negatives;
- accuracy, precision, recall, and F1;
- per-group detection rates;
- false-negative paths;
- feature-memory construction and scoring times.

AUROC and Average Precision remain useful future additions because they are threshold independent.

### Pixel-Level Evaluation

Future evaluation should add pixel-level AUROC, Average Precision, and threshold-dependent metrics where methodologically justified. Mask alignment and post-processing must be verified first.

### Qualitative Evaluation

Visual analysis should include representative normal images, false positives, true anomalies, false negatives, small defects, boundary defects, and heatmap localization. Examples must not be selected in a way that misrepresents overall behavior.

Dataset license restrictions must be checked before publishing source images, masks, or derived overlays.

## Recorded Exploratory Results

### Bottle at 224 × 224

| Aggregation | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum | 0.9398 | 1.0000 | 0.9206 | 0.9587 | 0 | 5 |
| Top 1% mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Top 5% mean | 0.9759 | 0.9692 | 1.0000 | 0.9844 | 2 | 0 |

### Capsule Reference at 320 × 320

| Metric | Value |
| --- | ---: |
| True positives | 104 |
| True negatives | 21 |
| False positives | 2 |
| False negatives | 5 |
| Accuracy | 0.9470 |
| Precision | 0.9811 |
| Recall | 0.9541 |
| F1 | 0.9674 |

These results document development evidence. They are not untouched blind benchmark claims because test images and results were inspected during iterative development.

## Experiment Management

Every meaningful experiment should record:

- experiment identifier and timestamp;
- Git revision;
- Python and dependency versions;
- dataset, category, and archive reference;
- split manifest and seed;
- backbone and pretrained weights;
- feature layers and input size;
- preprocessing parameters;
- memory fraction or coreset configuration;
- distance and chunking configuration;
- aggregation method;
- threshold method and value;
- metrics and grouped errors;
- runtime and memory measurements;
- output and artifact locations;
- warnings and known limitations.

Human-readable summaries should be backed by machine-readable experiment records. Machine-specific paths must not enter released artifacts.

## Reproducibility Strategy

Reproducibility requires:

- pinned direct dependencies;
- recorded Python version;
- deterministic file ordering and manifests;
- explicit random seeds;
- explicit preprocessing and model identifiers;
- isolated fitting, validation, and test roles;
- versioned experiment and artifact schemas;
- artifact checksums;
- clean-environment reconstruction tests.

The repository currently records Python 3.12.10, pinned dependencies, Bottle and Capsule manifests, sampling seeds, and typed artifact metadata. Complete experiment reports and checksums remain pending.

## Implemented Artifact Strategy

The current Python artifact contains:

```text
metadata.json
feature_memory.pt
```

Metadata includes dataset, category, backbone, input size, patch-grid size, embedding dimension, aggregation, top fraction, threshold, memory fraction, seed, and entry count.

The writer and loader validate tensor structure and metadata dimensions. The current format supports Python inference but is not a final public or cross-runtime package.

## Intended Cross-Runtime Export

A future evaluated package should contain:

```text
feature-extractor.onnx
feature-memory.<framework-neutral-format>
model-metadata.json
evaluation-summary.json
checksums.txt
license-and-attribution-notices/
```

Complete preprocessing semantics, pretrained-weight identity, feature layers, split identity, software versions, and evaluation context must be explicit.

The existing ONNX export proves feasibility but was created for the earlier technical spike. The selected 320 × 320 extractor must be exported and verified separately.

## Cross-Runtime Validation

Before a .NET backend consumes an artifact, fixed fixtures must compare:

- decoded RGB values and resized images;
- normalized tensors;
- ONNX feature outputs;
- patch embeddings;
- nearest-neighbor distances;
- patch and image scores;
- anomaly-map geometry;
- threshold decisions.

Numeric tolerances and version compatibility rules must be documented. Visual similarity alone is insufficient.

## Local Hardware Strategy

CPU execution remains the required baseline. The selected complete Capsule artifact provides individual predictions in approximately 1.44–1.46 seconds on the current machine. Artifact loading and extractor creation each take approximately 0.20 seconds in separate CLI processes and should be performed once in a persistent service.

The full validation and test evaluation is much slower because it scores many images against the complete memory. Hardware acceleration, indexing, and smarter memory reduction remain optional optimizations.

## Model Approval Gate

A candidate may become a documented reference model when:

- dataset license and attribution are recorded;
- validation and split isolation pass;
- fitting and threshold rules are reproducible;
- preprocessing is documented and reviewed;
- image-level metrics and grouped errors are recorded;
- runtime and memory are measured;
- artifact export and loading succeed;
- known limitations and test-set exposure are disclosed.

Pixel-level metrics and cross-runtime parity are still required before calling a model a complete cross-runtime release candidate.

## Public Release Gate

A model artifact may be released publicly only when:

- redistribution is permitted;
- required notices and citations are included;
- no datasets, private paths, credentials, or secrets are embedded;
- artifact and experiment schemas are versioned;
- checksums and compatibility checks pass;
- evaluation results are linked to the exact artifact;
- a Model Card documents intended use, limitations, and prohibited claims;
- Python/.NET parity is verified when .NET consumes it.

The current Capsule artifact is local and does not yet satisfy this public release gate.

## Failure Analysis Strategy

When performance is insufficient:

1. verify data integrity and partition isolation;
2. verify preprocessing and mask alignment;
3. inspect normal and anomalous score distributions;
4. review false positives and false negatives;
5. verify feature and anomaly-map geometry;
6. inspect threshold behavior;
7. measure feature-memory coverage and runtime;
8. adjust aggregation or scoring;
9. evaluate a coverage-preserving coreset or index;
10. compare a different backbone only after earlier causes are understood.

## Deferred Model Work

- all-category fitting;
- MVTec LOCO AD fitting;
- MVTec AD 2 private evaluation;
- supervised defect classification;
- backbone fine-tuning;
- autoencoder development;
- real-time camera and PLC integration;
- continual learning and automated retraining;
- distributed or multi-GPU training;
- production deployment optimization;
- real pharmaceutical product validation;
- regulatory validation;
- automated production accept/reject decisions.

## Completed Model-Development Milestones

1. Environment and dependency verification.
2. Dataset acquisition, checksums, extraction, and validation.
3. Deterministic Bottle and Capsule manifests.
4. Direct-resize preprocessing selection and configurable resolution.
5. Reusable feature extraction and patch embeddings.
6. Complete feature-memory creation.
7. Exact chunked nearest-neighbor scoring.
8. Configurable score aggregation and validation threshold.
9. Bottle evaluation and heatmaps.
10. Capsule generalization and resolution comparison.
11. Deterministic memory-sampling tradeoff evaluation.
12. Local model-artifact schema, writer, loader, and tests.
13. Export and verification of the 410.16 MiB Capsule artifact.
14. Correct normal and anomalous single-image CLI predictions.

## Immediate Next Steps

1. Consolidate the remaining specification and README documentation.
2. Add focused inference and artifact-compatibility tests.
3. Add machine-readable experiment summaries.
4. Implement pixel-level Capsule evaluation.
5. Define the complete framework-neutral artifact schema.
6. Export and verify the selected 320 × 320 ONNX extractor.
7. Select a framework-neutral feature-memory format.
8. Add artifact checksums and clean-environment reconstruction.
9. Build a minimal .NET console parity spike.

## Related Documentation

- `DevelopmentStatus.md` records verified status and immediate progress.
- `ArchitectureOverview.md` defines implemented and intended boundaries.
- `DatasetDocumentation.md` records dataset sources, licenses, and validation.
- `ProjectSpecification.md` defines functional and non-functional scope.
- A future `ModelCard.md` will document a released evaluated artifact.

## Last Updated

This strategy reflects the verified project state and selected Capsule reference configuration as of 2026-08-13.
