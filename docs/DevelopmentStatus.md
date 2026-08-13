# Industrial Visual Anomaly Detection – Development Status

## Document Purpose

This document records the current implementation status, verified technical findings, active development direction, open decisions, and immediate next steps for the Industrial Visual Anomaly Detection project.

It is a living status document. It distinguishes between:

- verified capabilities;
- selected implementation decisions;
- work currently in progress;
- planned but not yet implemented functionality.

The document must be updated whenever a relevant technical assumption, dataset decision, experiment result, or implementation milestone changes.

## Current Phase

The project is currently in the dataset-preparation and anomaly-baseline preparation phase.

Technical feasibility has already been demonstrated for:

- loading a pretrained ResNet18 backbone;
- extracting intermediate feature maps;
- combining multi-scale feature maps into patch embeddings;
- running the feature extractor on CPU;
- exporting the feature extractor to ONNX;
- validating the exported ONNX model;
- reproducing PyTorch outputs with ONNX Runtime;
- loading and inspecting real MVTec AD images;
- validating all three acquired MVTec datasets;
- creating a deterministic fitting and validation split for the first category.

No trained or fitted anomaly-detection model has been completed yet. No anomaly scores, thresholds, evaluation metrics, heatmaps, or production inference services have been implemented.

## Project Vision

The project is intended to become an industrial visual anomaly-detection system that:

- learns the visual appearance of normal products from defect-free images;
- detects images that deviate from the learned normal appearance;
- localizes suspicious image regions through anomaly maps;
- evaluates detection and localization quality against published benchmark data;
- exports the required neural feature extractor through ONNX;
- exposes inference through a client-neutral .NET backend;
- can later support both desktop and web clients.

The first implementation focuses on unsupervised anomaly detection. It must determine whether an image is normal or anomalous and indicate suspicious regions. It is not initially required to classify the precise defect type.

## Verified Development Environment

The following local environment has been verified:

| Component | Verified value |
| --- | --- |
| Operating system | Windows |
| Python | 3.12.10 |
| Virtual environment | `.venv` |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| Pillow | 12.2.0 |
| NumPy | 2.4.4 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.28.0 |
| ONNX Script | 0.7.1 |
| .NET SDK | 10.0.302 |
| Git | 2.55.0.windows.3 |

The Python version is recorded in `.python-version`. Direct Python dependencies are pinned in `requirements.txt`.

The Python source files currently compile successfully with `compileall`, and the installed Python packages pass `pip check`.

## Hardware Assessment

The development system uses an AMD Radeon 860M integrated GPU. The current PyTorch installation is CPU-only, and CUDA acceleration is unavailable.

This is acceptable for the initial project because:

- the selected ResNet18 backbone is comparatively lightweight;
- the first category contains a manageable number of images;
- feature extraction can be completed on CPU;
- initial experiments prioritize correctness and reproducibility over training speed;
- ONNX inference is intended to remain CPU-compatible.

The project must therefore avoid assuming CUDA availability. Optional hardware acceleration may be investigated later without becoming a requirement for local setup or inference.

## Verified PyTorch Capability

A pretrained ResNet18 model using the default TorchVision weights has been loaded and executed successfully.

For an artificial input tensor with shape:

```text
(1, 3, 224, 224)
```

the complete classification model produced an output with shape:

```text
(1, 1000)
```

This confirms that the local PyTorch and TorchVision installation can load and execute pretrained model weights.

## Verified CPU Performance

The complete pretrained ResNet18 model was benchmarked on an artificial input tensor after three warm-up executions. Twenty measured executions produced an observed average inference time of approximately 12–13 milliseconds on the current development system.

This measurement is only a technical reference. It does not represent end-to-end anomaly-detection performance because it excludes:

- image loading;
- production preprocessing;
- feature-memory construction;
- nearest-neighbor comparison;
- anomaly-map generation;
- threshold application;
- result serialization;
- client or API overhead.

No production latency claim is made at this stage.

## Verified Feature Extraction Strategy

The technical spike extracts intermediate ResNet18 features from `layer2` and `layer3`.

For an input tensor with shape `(1, 3, 224, 224)`, the verified feature shapes are:

| Feature output | Shape |
| --- | --- |
| ResNet18 `layer2` | `(1, 128, 28, 28)` |
| ResNet18 `layer3` | `(1, 256, 14, 14)` |

The `layer3` feature map is resized to the spatial dimensions of `layer2`. Both maps are concatenated along the channel dimension, producing:

```text
(1, 384, 28, 28)
```

The combined feature map is rearranged into one embedding per spatial position:

```text
(784, 384)
```

This means that one 224 × 224 input image currently produces 784 local patch embeddings, each containing 384 feature values.

This behavior is suitable for a PatchCore-style anomaly-detection baseline, but the final baseline implementation has not yet been completed.

## Verified ONNX Export

The custom ResNet18 feature extractor has been exported successfully to ONNX with a fixed input shape of:

```text
(1, 3, 224, 224)
```

The provisional files produced by the export are:

```text
resnet18_feature_extractor.onnx
resnet18_feature_extractor.onnx.data
```

The exported model passed the official ONNX model checker.

The generated files are ignored by Git because they are development artifacts. They must not be treated as released model artifacts until preprocessing, metadata, licensing, versioning, evaluation, and packaging have been finalized.

## Verified PyTorch and ONNX Consistency

The same artificial image tensor has been processed by both the PyTorch feature extractor and ONNX Runtime using the CPU execution provider.

The observed maximum absolute differences were:

| Output | Maximum difference |
| --- | ---: |
| `layer2` | 0.00000241 |
| `layer3` | 0.00000161 |

These differences are sufficiently small for the technical proof of concept. They show that the exported ONNX model reproduces the PyTorch feature outputs with effectively equivalent numerical results for the verified input.

## Verified Preprocessing

The preprocessing transform associated with the default pretrained ResNet18 weights has been executed successfully on a real MVTec AD bottle image.

The inspected source image had the following properties:

| Property | Value |
| --- | --- |
| Original mode | RGB |
| Original size | 900 × 900 pixels |
| Resulting tensor shape | `(3, 224, 224)` |
| Tensor data type | `torch.float32` |

The verified transformation applies:

- conversion to RGB;
- resizing to 256 pixels according to the TorchVision preset;
- a 224 × 224 center crop;
- bilinear interpolation;
- normalization with the ImageNet mean and standard deviation expected by the pretrained weights.

The script confirms the technical preprocessing path. A visual review of the resulting crop must still be added before the preprocessing contract is finalized.

## Dataset Storage

Datasets are stored outside the Git repository under:

```text
C:/dev/data/industrial-visual-anomaly-detection/
```

The local storage is separated into:

```text
archives/
raw/
```

Dataset archives, extracted images, masks, generated validation reports, model artifacts, and other large derived files must not be committed to the public source repository.

Only source code, configuration, deterministic split manifests, documentation, and small explicitly permitted examples may be versioned.

## Dataset Acquisition and Validation Status

Three datasets have been downloaded, archived locally, extracted, inspected, and validated.

### MVTec AD

| Property | Verified value |
| --- | --- |
| Archive | `mvtec_anomaly_detection.tar.xz` |
| SHA-256 | `CF4313B13603BEC67ABB49CA959488F7EEDCE2A9F7795EC54446C649AC98CD3D` |
| Categories | 15 |
| Normal training images | 3,629 |
| Normal test images | 467 |
| Anomalous test images | 1,258 |
| Ground-truth masks | 1,258 |
| Total PNG files | 6,612 |

All expected categories and directory structures are present. All PNG files are readable. Every anomalous test image has a corresponding mask, and all 1,258 image-mask pairs passed validation.

### MVTec LOCO AD

| Property | Verified value |
| --- | --- |
| Archive | `mvtec_loco_anomaly_detection.tar.xz` |
| SHA-256 | `9E7C84DBA550FD2E59D8E9E231C929C45BA737B6B6A6D3814100F54D63AAE687` |
| Categories | 5 |
| Normal training images | 1,778 |
| Normal validation images | 305 |
| Normal test images | 575 |
| Logical anomaly test images | 561 |
| Structural anomaly test images | 432 |
| Mask groups | 993 |
| Mask files | 1,246 |
| Total PNG files | 4,897 |

All PNG files are readable. All 993 anomalous test images have matching non-empty mask groups, and all 1,246 mask files passed validation. Positive mask values are validated against the category-specific definitions in `defects_config.json` rather than being treated as universally binary.

The local extracted image count differs from a published aggregate count by seven images. The difference is isolated to `splicing_connectors`, whose local training and validation folders contain six and one additional images respectively. The local archive checksum is recorded, and no files have been removed to force agreement with an external count.

### MVTec AD 2

| Property | Verified value |
| --- | --- |
| Archive | `mvtec_ad_2.tar.gz` |
| SHA-256 | `C0DED99EF32BFC8E352D52BEB44515E5B292B8598CB963AADFA91CA0763505E4` |
| Categories | 8 |
| Normal training images | 2,528 |
| Normal validation images | 302 |
| Public normal test images | 379 |
| Public anomalous test images | 705 |
| Public masks | 705 |
| Private test images | 2,045 |
| Private mixed test images | 2,045 |
| Total PNG files | 8,709 |

All PNG files are readable. All 705 public anomalous images have matching masks, and the public image-mask pairs passed content validation.

Ground truth for the private test partitions is not included locally. Evaluation of those partitions would require the official external evaluation mechanism and is not part of the initial baseline.

## Machine-Readable Dataset Reports

All three dataset validators now support an optional `--report` argument and can write schema-versioned JSON reports after successful validation.

The generated reports contain:

- dataset identifier and resolved local root;
- discovered categories;
- per-category and aggregate inventories;
- image dimensions and modes;
- validated mask, mask-group, or image-mask pair counts;
- individual validation-stage results.

The reports currently use schema version `1`. They are written only after every implemented validation stage passes. Failed validation terminates without producing a misleading successful report.

Generated files are stored under `validation-reports/`, contain machine-specific local paths, and are excluded from Git.

## Selected Initial Dataset and Category

The first anomaly-detection baseline will use:

| Setting | Selected value |
| --- | --- |
| Dataset | MVTec AD |
| Category | `bottle` |
| Training source | `bottle/train/good` |
| Official evaluation source | complete `bottle/test` partition |
| Localization ground truth | `bottle/ground_truth` |

The bottle category was selected because:

- it has a manageable dataset size;
- its normal images have a consistent background, position, and illumination;
- the defects are visually recognizable;
- pixel-level masks are available for anomalous test images;
- it provides a clear first case for understanding the complete pipeline.

The model will initially learn only from normal bottle images. The official defect folder names are retained for grouped evaluation and analysis, not used as supervised training labels.

## Deterministic Bottle Split

The 209 normal images in `bottle/train/good` have been divided deterministically into:

| Partition | Image count | Purpose |
| --- | ---: | --- |
| Fitting | 167 | Build the normal feature memory |
| Normal validation | 42 | Select and verify the anomaly threshold without using the official test set |

The split uses random seed `42` and has no overlap. All 209 source images are represented exactly once.

The resulting manifest is versioned at:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
```

The official `bottle/test` partition remains untouched for final evaluation.

## Current Model Strategy

The selected first baseline is a PatchCore-style anomaly-detection pipeline built around a pretrained ResNet18 feature extractor.

The planned sequence is:

```text
normal fitting images
→ deterministic preprocessing
→ pretrained ResNet18 feature extraction
→ multi-scale patch embeddings
→ normal feature memory
→ nearest-neighbor comparison for a new image
→ patch-level anomaly scores
→ image-level anomaly score
→ anomaly map
→ threshold-based normal/anomalous decision
```

The pretrained backbone is not initially fine-tuned. The first objective is to measure how well pretrained visual features represent normal industrial appearance.

The implementation should remain understandable and explicit. Core steps such as preprocessing, embedding creation, feature-memory construction, scoring, threshold selection, and evaluation should not be hidden behind a large external anomaly-detection framework.

## Current Evaluation Strategy

The first baseline will be evaluated on the complete official MVTec AD bottle test partition.

The intended evaluation includes:

- image-level anomaly scores;
- a threshold-derived normal/anomalous decision;
- image-level classification metrics;
- pixel-level anomaly maps;
- pixel-level localization metrics using the supplied masks;
- grouped results by official bottle defect folder;
- representative visual result examples.

The threshold must be derived from the 42-image normal validation partition and not from the official test labels.

The exact threshold method and final metric set remain open decisions. Candidate metrics include image-level AUROC, pixel-level AUROC, average precision, and threshold-dependent precision, recall, and F1 score.

No evaluation results exist yet.

## Current Architecture Direction

The intended long-term architecture separates model development from application delivery:

```text
Python model development
→ evaluated and versioned artifacts
→ ONNX feature extractor and model metadata
→ client-neutral .NET inference backend
→ desktop client and/or web client
```

Python owns:

- dataset validation;
- preprocessing experiments;
- fitting the anomaly-detection baseline;
- evaluation;
- threshold selection;
- artifact export.

The future .NET backend is intended to own:

- artifact loading;
- request validation;
- production preprocessing consistent with Python;
- ONNX inference;
- anomaly scoring and threshold application;
- stable result contracts;
- access for multiple client types.

No .NET backend, desktop client, or web client has been implemented yet.

## Current Repository Contents

The repository currently contains:

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

The scripts currently provide:

- technical feature-extractor and ONNX checks;
- MVTec AD validation with optional JSON reporting;
- MVTec LOCO AD validation with category-specific mask-value checks and optional JSON reporting;
- MVTec AD 2 validation with optional JSON reporting;
- deterministic MVTec AD training splits;
- preprocessing inspection for individual images.

The local virtual environment, editor settings, datasets, generated ONNX files, caches, reports, and other generated artifacts are excluded by `.gitignore`.

## Intended Model Artifact Contract

The final artifact contract has not yet been implemented. It is expected to contain more than a standalone ONNX file.

A future versioned artifact package will likely include:

```text
feature-extractor.onnx
feature-memory artifact
model-metadata.json
preprocessing.json
thresholds.json
evaluation-summary.json
```

The metadata should identify at least:

- model and artifact versions;
- pretrained backbone and weight version;
- input dimensions;
- preprocessing configuration;
- selected feature layers;
- embedding dimensions;
- dataset and category;
- deterministic split manifest;
- feature-memory configuration;
- threshold method and value;
- evaluation results;
- framework and dependency versions;
- license and attribution information.

## Open Decisions

The following decisions remain open:

- Which visual crop inspection method should be added to verify center cropping?
- Should the baseline use all fitting embeddings or a reduced coreset?
- Which coreset selection method and sampling ratio should be used if reduction is enabled?
- Which nearest-neighbor implementation should be used during Python development?
- How should patch-level scores be converted into the final image-level score?
- How should anomaly maps be resized and smoothed?
- Which normal-validation threshold method should be selected?
- Which evaluation metrics form the required MVP result set?
- Which experiment-report format should become versioned?
- Which parts of anomaly scoring should later run in ONNX, managed .NET, or a dedicated numerical library?
- How should model and feature-memory artifacts be packaged for release?

The initial dataset, category, backbone family, feature layers, input size, and deterministic split are no longer open decisions for the first baseline.

## Deferred Work

The following work is intentionally deferred until the first Python anomaly baseline has been fitted and evaluated:

- a production .NET inference backend;
- desktop and web clients;
- database persistence;
- authentication and authorization;
- deployment packaging;
- GPU-specific optimization;
- fine-tuning the pretrained backbone;
- multi-category model management;
- defect-type classification;
- evaluation through the private MVTec AD 2 benchmark service;
- production monitoring and model-drift handling.

## Immediate Next Steps

The next implementation steps are:

1. Add a visual preprocessing inspection that saves the original, resized, and cropped Bottle image for comparison.
2. Refactor the technical feature extractor from `environment_check.py` into reusable Python modules.
3. Implement deterministic loading of the Bottle fitting and validation partitions from the split manifest.
4. Add automated tests for dataset report generation and schema contents.
5. Extract patch embeddings for the 167 fitting images.
6. Build the first normal feature memory.
7. Measure feature-memory size and exact nearest-neighbor runtime.
8. Implement nearest-neighbor anomaly scoring.
9. Score the 42 normal validation images and define the first threshold method.
10. Evaluate the baseline on the official Bottle test partition.
11. Generate image-level metrics, anomaly maps, and representative result visualizations.
12. Record the experiment configuration and results reproducibly.

The machine-readable dataset-report milestone is complete. The visual preprocessing inspection is the next active implementation step.

## Last Verified Status

As of 2026-08-13:

- the Python 3.12 virtual environment is operational;
- pinned project dependencies are recorded;
- the source files compile successfully;
- the pretrained ResNet18 backbone runs on CPU;
- intermediate multi-scale features and patch embeddings can be extracted;
- the provisional feature extractor exports successfully to ONNX;
- ONNX Runtime reproduces the PyTorch feature outputs with very small numerical differences;
- the pretrained TorchVision preprocessing transform runs successfully on a real bottle image;
- MVTec AD, MVTec LOCO AD, and MVTec AD 2 are downloaded and stored outside the repository;
- the archive checksums are recorded;
- all three dataset validators can generate schema-versioned JSON reports after successful validation;
- MVTec LOCO AD mask values are validated against each category's `defects_config.json`;
- MVTec AD bottle is selected as the first implementation category;
- the normal bottle training images have been split deterministically into 167 fitting and 42 validation images;
- the split manifest is versioned and contains no overlap;
- no anomaly feature memory has been fitted yet;
- no anomaly threshold has been selected;
- no official bottle test evaluation has been performed;
- no production model artifact has been released;
- no .NET backend or user-facing client has been implemented.

The project has therefore completed environment verification, technical feature-extraction feasibility, dataset acquisition, dataset validation, machine-readable validation reporting, initial category selection, and deterministic split preparation. The next active milestone is visual verification of the selected Bottle preprocessing before the reusable anomaly-detection modules are implemented.