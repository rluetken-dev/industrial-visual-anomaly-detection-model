# Industrial Visual Anomaly Detection – Model Development Strategy

## Purpose

This document defines the strategy for developing, fitting, validating, evaluating, comparing, exporting, serving, and evolving anomaly-detection models for the Industrial Visual Anomaly Detection project.

It is intended to prevent:

- test-set leakage;
- irreproducible experiments;
- misleading benchmark claims;
- undocumented model changes;
- incompatible model artifacts;
- divergence between offline and service inference;
- premature optimization;
- confusion between benchmark performance and production readiness.

The first end-to-end model-development cycle is complete. Bottle established the initial pipeline, and Capsule tested generalization, input resolution, memory sampling, artifact export, reusable inference, and integration through a persistent Python service and ASP.NET Core backend.

## Strategy Status

This document distinguishes between:

- **Implemented and verified** – demonstrated through executable code, automated tests, or recorded experiments;
- **Selected reference strategy** – used by the current reference artifact and service;
- **Exploratory result** – useful development evidence that is not an untouched blind benchmark;
- **Open decision** – requires further implementation or evaluation;
- **Deferred work** – excluded from the current development cycle.

## Implemented and Verified Capabilities

- CPU-based PyTorch and TorchVision execution;
- pretrained frozen ResNet18 feature extraction;
- `layer2` and `layer3` feature fusion;
- configurable square preprocessing at 224 x 224 and 320 x 320;
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
- artifact writing, validation, and loading;
- file-path and binary-stream inference;
- prediction CLI;
- persistent FastAPI inference service;
- startup-time artifact and feature-extractor initialization;
- internal liveness and multipart prediction endpoints;
- ASP.NET Core integration through HTTP;
- manually verified path/stream parity and end-to-end inference;
- provisional ONNX export and earlier PyTorch/ONNX numerical parity;
- 65 passing automated Python tests.

These capabilities establish a functioning reference implementation and application integration boundary. They do not establish production readiness, regulatory validation, real-world transfer, deployment hardening, or localization accuracy.

## Reference Model Configuration

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
| Input size | 320 x 320 |
| Preprocessing | RGB, direct bilinear resize with antialiasing, tensor conversion, ImageNet normalization |
| Embedding dimension | 384 |
| Patch grid | 40 x 40 |
| Feature-memory entries | 280,000 |
| Feature-memory fraction | 1.0 |
| Distance search | Exact chunked nearest neighbor |
| Image aggregation | Mean of highest 1% of patch scores |
| Threshold | Maximum normal-validation score |
| Stored threshold | 2.501821517944336 |
| Inference runtime | Python/PyTorch |
| Service boundary | Internal FastAPI HTTP service |

The split manifest is:

```text
configs/splits/mvtec-ad-capsule-seed-42.json
```

The model detects deviation from normal appearance and returns a spatial patch-score map. It does not classify the exact defect type.

## Model Development Principles

1. Verify dataset source, license, structure, and integrity before use.
2. Use only normal fitting images to construct the normal reference memory.
3. Keep threshold selection independent from official test labels.
4. Preserve preprocessing and scoring semantics across fitting, CLI, and service inference.
5. Begin with explicit, testable mechanics before advanced optimization.
6. Record every meaningful model change as a reproducible experiment.
7. Separate image-level detection from pixel-level localization claims.
8. Review false positives and false negatives in addition to aggregate metrics.
9. Treat runtime and memory consumption as model-quality constraints.
10. Keep artifacts inseparable from preprocessing, backbone, dimensions, aggregation, and threshold metadata.
11. Mark results as exploratory when test data influenced later decisions.
12. Keep public API concerns outside the model implementation.
13. Verify offline and service inference against the same artifact.
14. Never present benchmark results as evidence of production suitability.

## Evidence-Gated Development Process

```text
dataset qualification                         completed
        |
        v
data validation and deterministic manifests   completed
        |
        v
preprocessing verification                    completed
        |
        v
reusable feature extraction                   completed
        |
        v
complete feature memory                       completed
        |
        v
exact anomaly scoring                         completed
        |
        v
normal-validation threshold                   completed
        |
        v
Bottle baseline evaluation                    completed
        |
        v
Capsule generalization and resolution study   completed
        |
        v
memory-sampling trade-off study               completed
        |
        v
artifact export and CLI inference             completed
        |
        v
stream inference and service integration      completed
        |
        v
ASP.NET Core end-to-end request               completed
        |
        v
service hardening and readiness               pending
        |
        v
pixel-level evaluation                        pending
        |
        v
deployment packaging                          pending
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
- **Ground-truth masks:** evaluate localization when pixel-level metrics are implemented.

Test labels must not construct memory or determine the threshold. Because Bottle and Capsule results were inspected and informed later development, current comparisons are exploratory rather than untouched final benchmarks.

## Preprocessing Strategy

The deterministic preprocessing is:

```text
decode image
-> convert to RGB
-> direct resize to configured square input
-> convert to torch.float32 tensor
-> normalize with ImageNet mean and standard deviation
```

Bottle uses 224 x 224. Capsule uses 320 x 320 because higher resolution materially improved sensitivity to small defects.

The default TorchVision center-crop path was rejected for Bottle because it removed part of the object boundary. Direct resizing introduces no aspect-ratio distortion for the inspected square Bottle and Capsule images.

Non-square categories require a separate documented choice between direct resizing, aspect-ratio preservation, and padding.

No random augmentation belongs to the current reference model. Future augmentation must preserve the meaning of normal data, use explicit seeds, and retain deterministic evaluation preprocessing.

## Feature Extraction Strategy

The frozen ResNet18 backbone supplies `layer2` and `layer3` features. `layer3` is bilinearly resized to the spatial resolution of `layer2`, concatenated along the channel dimension, and rearranged into 384-dimensional patch embeddings.

At 224 x 224:

```text
layer2:          (1, 128, 28, 28)
layer3:          (1, 256, 14, 14)
patch embeddings:      (784, 384)
```

At 320 x 320:

```text
layer2:          (1, 128, 40, 40)
layer3:          (1, 256, 20, 20)
patch embeddings:     (1600, 384)
```

Backbone replacement or fine-tuning is deferred until data, preprocessing, scoring, threshold behavior, runtime, and failure cases have been examined first.

## Feature Memory Strategy

The feature memory concatenates embeddings from all normal fitting images in deterministic loader order.

It is bound to:

- dataset and category;
- split manifest;
- input and preprocessing configuration;
- backbone and pretrained weights;
- selected feature layers;
- embedding construction and dimension;
- sampling or coreset configuration;
- framework and artifact versions.

Incompatible feature memories must never be combined.

The Capsule reference memory contains:

```text
175 x 1600 = 280000 embeddings
shape: (280000, 384)
dtype: float32
size: approximately 410.16 MiB
```

The current serialization uses `torch.save` and is treated as trusted deployment input. A framework-neutral format is optional future portability work; it is not required by the implemented HTTP integration because Python loads the artifact.

## Memory-Reduction Strategy

Complete memory is established first as the quality reference. Optimization is evaluated against that fixed configuration.

| Fraction | Entries | Validation + test scoring | Recall | F1 | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 280,000 | 225.06 s | 0.9541 | 0.9674 | 5 |
| 75% | 210,000 | 184.33 s | 0.8991 | 0.9423 | 11 |
| 50% | 140,000 | 120.36 s | 0.8624 | 0.9216 | 15 |
| 25% | 70,000 | 64.15 s | 0.6330 | 0.7753 | 40 |

Random sampling reduces runtime but loses too much feature coverage. The reference artifact therefore retains 100% of memory. Future reduction should use a coverage-preserving coreset or another method evaluated against the complete-memory reference.

## Nearest-Neighbor Strategy

The implementation uses exact Euclidean nearest-neighbor distance. Reference memory is processed in configurable chunks to control temporary memory without changing results.

Recorded scoring configuration must include:

- distance definition;
- query and memory chunking;
- tensor data type;
- patch-grid dimensions;
- memory size;
- runtime environment.

Approximate search may be considered if it preserves an explicit quality target and is compared with fixed exact-search results.

## Image-Level Scoring Strategy

Supported aggregation methods are:

- maximum patch distance;
- mean of the highest configurable fraction of patch distances.

Top-one-percent mean is selected because it was more robust than one extreme patch and performed strongly in the explored Bottle and Capsule evaluations.

Aggregation parameters belong to artifact metadata and must not change silently during service inference.

## Threshold Strategy

The reference threshold is the maximum image score among normal validation images:

```text
threshold = max(normal validation scores)
```

Only scores strictly greater than the threshold are anomalous.

This constrains false positives on the validation-normal distribution but cannot directly optimize anomaly recall. Alternative thresholds must be selected without final test labels from the experiment being reported.

The Capsule artifact stores `2.501821517944336`.

## Anomaly-Map Strategy

Nearest-neighbor distances are reconstructed as a patch grid. Visualization can resize, normalize, colorize, and blend the map with the image.

Fixed threshold-based normalization is preferred for comparison across images. Heatmap colors are presentation data and must not replace raw scores.

Quantitative localization metrics are not implemented. Any smoothing, normalization, or pixel threshold that affects metrics must be recorded as experiment configuration.

## Evaluation Strategy

### Image-Level Evaluation

Implemented evaluation reports:

- normal and per-group score distributions;
- true positives, true negatives, false positives, and false negatives;
- accuracy, precision, recall, and F1;
- per-group detection rates;
- false-negative paths;
- feature-memory construction and scoring times.

AUROC and Average Precision remain useful future additions because they are threshold independent.

### Pixel-Level Evaluation

Future work should add pixel-level AUROC, Average Precision, and justified threshold-dependent metrics. Mask alignment and post-processing must be verified first.

### Qualitative Evaluation

Visual analysis should include representative normal images, false positives, true anomalies, false negatives, small defects, boundary defects, and heatmaps. Examples must not be selected in a way that misrepresents overall behavior.

Dataset license restrictions must be checked before publishing images, masks, or overlays.

## Recorded Exploratory Results

### Bottle at 224 x 224

| Aggregation | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maximum | 0.9398 | 1.0000 | 0.9206 | 0.9587 | 0 | 5 |
| Top 1% mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Top 5% mean | 0.9759 | 0.9692 | 1.0000 | 0.9844 | 2 | 0 |

### Capsule Reference at 320 x 320

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

- experiment identifier, timestamp, and Git revision;
- Python and dependency versions;
- dataset, category, archive reference, split manifest, and seed;
- backbone, pretrained weights, feature layers, and input size;
- preprocessing parameters;
- memory fraction or coreset configuration;
- distance and chunking configuration;
- aggregation method;
- threshold method and value;
- metrics and grouped errors;
- runtime and memory measurements;
- output and artifact locations;
- warnings and known limitations.

Human-readable summaries should be backed by machine-readable records. Machine-specific paths must not enter released artifacts.

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

The repository records Python 3.12.10, pinned dependencies, Bottle and Capsule manifests, sampling seeds, and typed artifact metadata. Complete experiment reports and checksums remain pending.

## Artifact Strategy

The current Python artifact contains:

```text
metadata.json
feature_memory.pt
```

Metadata includes dataset, category, backbone, input size, patch-grid size, embedding dimension, aggregation, top fraction, threshold, memory fraction, seed, and entry count.

The writer and loader validate tensor structure and metadata dimensions. The current format supports trusted Python inference and the implemented service.

Future artifact evolution may add full preprocessing metadata, pretrained-weight identity, checksums, evaluation summaries, dependency versions, and license notices. ONNX and framework-neutral memory storage remain optional portability improvements.

## Offline and Service Inference Strategy

`predict_image` provides path-based CLI inference. `predict_image_stream` provides binary-stream inference for HTTP uploads. Both use the same artifact, preprocessing, extractor, scoring, aggregation, and threshold logic.

Manual parity verification for the same image produced identical score, threshold, and decision:

```text
Path score:       1.848755
Stream score:     1.848755
Score difference: 0.000000000000
Same threshold:   True
Same decision:    True
```

Future model changes must preserve this shared path and include regression coverage appropriate to the changed behavior.

## Inference-Service Strategy

The internal FastAPI service is the selected model boundary. At startup it loads the configured artifact and creates the feature extractor once. Requests reuse an `InferenceRuntime` stored in application state.

Current configuration:

```text
IVAD_MODEL_ARTIFACT
IVAD_MEMORY_CHUNK_SIZE
```

Current endpoints:

```text
GET  /health/live
POST /api/v1/predictions
```

The service response contains model ID, category, score, threshold, and Boolean anomaly decision. The artifact directory name currently supplies the model ID.

Prediction execution is serialized with a process-local lock. This conservative baseline protects the shared runtime and bounds CPU and memory pressure. Parallel execution must not be enabled without measurement. Multiple service workers independently load the feature memory and multiply RAM use.

The service is internal. ASP.NET Core owns public upload limits, client-facing validation, Problem Details, trace identifiers, and stable external contracts. Python should still gain defense-in-depth malformed-image validation and structured internal errors.

## Backend Integration Strategy

ASP.NET Core communicates with FastAPI over HTTP. This direction was selected instead of embedding Python in the .NET process or immediately reimplementing the full pipeline in .NET.

Reasons include:

- preservation of verified PyTorch behavior;
- one persistent copy of the large model state per service process;
- isolation of Python dependencies and failures;
- a language-neutral boundary;
- independent model and backend versioning;
- reduced need for premature numerical parity work.

A real Capsule request was verified through:

```text
ASP.NET Core POST /api/v1/analyses
-> FastAPI POST /api/v1/predictions
-> Python/PyTorch inference
-> ASP.NET Core response
```

The verified anomalous score was `4.992109298706055` against threshold `2.501821517944336`.

ONNX or native .NET inference may be reconsidered only when a concrete portability, latency, packaging, or offline requirement justifies the additional implementation and parity work.

## Local Hardware Strategy

CPU execution remains required. Direct Capsule predictions took approximately 1.44–1.46 seconds on the current machine. The verified backend-to-service request reported approximately 1.8 seconds total processing time.

Artifact loading and extractor creation are performed once at service startup. Full validation and test evaluation remain slower because many images are scored against complete memory.

Hardware acceleration, indexing, smarter memory reduction, and concurrency changes remain optional optimizations that require measured quality and resource comparisons.

## Model Approval Gate

A candidate may become a documented reference model when:

- dataset license and attribution are recorded;
- validation and split isolation pass;
- fitting and threshold rules are reproducible;
- preprocessing is documented and reviewed;
- image-level metrics and grouped errors are recorded;
- runtime and memory are measured;
- artifact export and loading succeed;
- offline and service inference agree for fixed verification inputs;
- known limitations and test-set exposure are disclosed.

Pixel-level metrics are required before claiming evaluated localization quality, but not before using patch maps as qualitative explanation aids.

## Public Release Gate

A model artifact may be released publicly only when:

- redistribution is permitted;
- required notices and citations are included;
- no datasets, private paths, credentials, or secrets are embedded;
- artifact and experiment schemas are versioned;
- checksums and compatibility checks pass;
- evaluation results link to the exact artifact;
- a Model Card documents intended use, limitations, and prohibited claims;
- the supported runtime and service versions are explicit.

Python/.NET numerical parity is required only if a future .NET runtime directly consumes model tensors. It is not required for the current HTTP integration, where Python remains authoritative.

The Capsule artifact is local and does not yet satisfy this public release gate.

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
10. compare another backbone only after earlier causes are understood.

Service failures must additionally be separated into configuration, artifact loading, image decoding, model inference, timeout, transport, and response-contract failures.

## Deferred Model Work

- all-category fitting;
- MVTec LOCO AD fitting;
- MVTec AD 2 private evaluation;
- supervised defect classification;
- backbone fine-tuning;
- autoencoder development;
- continual learning and automated retraining;
- distributed or multi-GPU training;
- real-time camera and PLC integration;
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
11. Deterministic memory-sampling trade-off evaluation.
12. Local artifact schema, writer, loader, and tests.
13. Export and verification of the 410.16 MiB Capsule artifact.
14. Correct normal and anomalous CLI predictions.
15. Binary-stream inference with verified path parity.
16. Persistent FastAPI runtime and prediction endpoint.
17. Successful ASP.NET Core-to-Python end-to-end prediction.

## Immediate Next Steps

1. Add service readiness based on initialized runtime state.
2. Define structured service errors and malformed-image handling.
3. Add defense-in-depth content validation at the Python boundary.
4. Add focused inference and artifact-compatibility regression tests.
5. Define timeout, cancellation, retry, and concurrency policies.
6. Add structured service timing and failure logging.
7. Create a reproducible local startup workflow for both processes.
8. Add machine-readable experiment summaries.
9. Implement pixel-level Capsule evaluation.
10. Add artifact checksums and clean-environment reconstruction.

## Related Documentation

- `DevelopmentStatus.md` records verified status and immediate progress.
- `ArchitectureOverview.md` defines implemented and intended boundaries.
- `DatasetDocumentation.md` records dataset sources, licenses, and validation.
- `ProjectSpecification.md` defines functional and non-functional scope.
- A future `ModelCard.md` will document a released evaluated artifact.

## Last Updated

This strategy reflects the verified project state and selected Capsule reference configuration as of 2026-08-14.
