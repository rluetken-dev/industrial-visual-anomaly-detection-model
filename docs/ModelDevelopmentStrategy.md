# Industrial Visual Anomaly Detection – Model Development Strategy

## Purpose

This document defines the current strategy for developing, fitting, validating, evaluating, comparing, and exporting anomaly-detection models for the Industrial Visual Anomaly Detection project.

Its purpose is to prevent:

- accidental test-set leakage;
- irreproducible experiments;
- misleading metrics;
- undocumented model changes;
- premature optimization;
- confusion between benchmark performance and production readiness.

The first dataset, category, split, backbone, feature layers, and input size have now been selected. No anomaly feature memory has yet been fitted, and no anomaly-detection result has yet been evaluated.

## Strategy Status

This document distinguishes between:

- **Verified capability** – demonstrated successfully through executable code or dataset validation;
- **Selected strategy** – fixed for the first baseline unless measured evidence requires a documented revision;
- **Open decision** – to be resolved through validation evidence;
- **Deferred work** – excluded from the first model-development cycle.

## Verified Capabilities

The current technical and dataset work has verified:

- CPU-based PyTorch and TorchVision execution;
- loading pretrained ResNet18 weights;
- extracting `layer2` and `layer3` feature maps;
- combining multi-scale feature maps;
- creating local patch embeddings;
- exporting a feature-extractor wrapper to ONNX;
- validating the ONNX graph;
- executing the model through ONNX Runtime;
- close numerical agreement between PyTorch and ONNX outputs;
- sufficient local CPU performance for continued proof-of-concept development;
- dataset structure, inventory, readability, and available mask validation for MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- deterministic direct 224 × 224 preprocessing of real normal and anomalous MVTec AD Bottle images;
- visual comparison of direct resizing with the default TorchVision center-crop pipeline;
- deterministic creation of the initial Bottle fitting and validation split;
- complete split coverage with no overlap.

These capabilities demonstrate that the development environment, datasets, preprocessing path, feature extraction, and ONNX boundary are viable.

They do not yet demonstrate:

- anomaly detection quality;
- localization quality;
- a fitted feature memory;
- threshold quality;
- complete experiment reproducibility;
- production preprocessing parity with .NET;
- production readiness.

## Selected First Baseline

The first anomaly-detection baseline uses:

| Setting | Selected value |
| --- | --- |
| Dataset | MVTec AD |
| Category | `bottle` |
| Learning mode | Normal-only, unsupervised anomaly detection |
| Fitting source | `bottle/train/good` split manifest entries |
| Fitting images | 167 |
| Normal validation images | 42 |
| Final evaluation source | Complete official `bottle/test` partition |
| Localization ground truth | `bottle/ground_truth` |
| Random seed | 42 |
| Backbone | ResNet18 with default pretrained TorchVision weights |
| Feature layers | `layer2` and `layer3` |
| Input size | 224 × 224 pixels |
| Preprocessing | RGB conversion, direct 224 × 224 bilinear resize with antialiasing, tensor conversion, and ImageNet normalization |
| Backbone training | Frozen for the first baseline |
| Model family | PatchCore-style feature-memory approach |

The split membership is recorded in:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
```

The selected model detects whether an image differs from learned normal appearance and localizes suspicious regions. It does not initially classify the exact defect type.

## Model Development Principles

Model development follows these principles:

1. Dataset licenses, structures, and integrity are verified before use.
2. One category is used to understand and validate the complete first pipeline.
3. The official test partition remains isolated until final evaluation.
4. Threshold selection uses only the normal validation partition or another explicitly approved non-test method.
5. Preprocessing must be equivalent across fitting, evaluation, ONNX Runtime, and future .NET inference.
6. The pretrained backbone remains frozen for the first baseline.
7. The simplest correct feature-memory implementation is measured before optimization.
8. Every reported result must be linked to a reproducible experiment configuration.
9. Model decisions use validation evidence rather than final test performance.
10. Image-level and pixel-level performance are reported separately.
11. Quantitative metrics are supplemented with qualitative error analysis.
12. Exported artifacts contain sufficient metadata to reproduce inference.
13. CPU runtime and memory usage are treated as model-quality constraints.
14. Benchmark performance is not presented as evidence of production readiness.

## Development Phases

The development process is divided into evidence gates:

```text
Dataset qualification                         completed
        ↓
Data pipeline and split validation             completed
        ↓
Real-image preprocessing verification          completed
        ↓
Reusable feature-extraction modules            next
        ↓
Initial full feature memory
        ↓
Nearest-neighbor anomaly scoring
        ↓
Normal-validation threshold selection
        ↓
Locked official test evaluation
        ↓
Optimization and optional coreset comparison
        ↓
Versioned artifact export
        ↓
Python/ONNX/.NET parity validation
```

Each phase must produce sufficient evidence before the next phase becomes part of the implementation baseline.

## Phase 1: Dataset Qualification

Dataset qualification has been completed for:

- MVTec AD;
- MVTec LOCO AD;
- MVTec AD 2.

The recorded qualification includes:

- official source and attribution;
- license and redistribution restrictions;
- archive name and SHA-256 checksum;
- extracted directory structure;
- categories and partitions;
- image and mask counts;
- image dimensions and modes;
- image readability;
- image-mask correspondence;
- available mask-content validation;
- suitability for normal-only fitting.

MVTec AD `bottle` was selected for the first baseline because it provides:

- a clear normal state;
- a consistent background, object position, and illumination;
- visually understandable anomalies;
- 209 normal training images;
- labeled normal and anomalous test images;
- pixel-level masks for anomalous test images;
- a manageable CPU-oriented first experiment.

MVTec LOCO AD and MVTec AD 2 remain qualified future evaluation datasets. They are not part of the first baseline.

## Phase 2: Data Pipeline Validation

The implemented validation scripts check relevant combinations of:

- configured dataset root existence;
- expected categories;
- required partitions and root files;
- deterministic file discovery;
- image and mask inventories;
- readable PNG files;
- image dimensions and modes;
- required masks;
- image-mask filename correspondence;
- mask dimensions, modes, and valid content.

The Bottle split was separately checked for:

- 209 source images;
- 167 fitting entries;
- 42 validation entries;
- no overlap;
- complete source coverage.

All three dataset validators can now generate schema-versioned JSON reports after successful validation. The reports remain ignored local output because they contain resolved machine-specific dataset paths. Duplicate-content detection may be added later and must not change the already recorded deterministic Bottle split silently.

## Data Partition Strategy

The partition roles are fixed for the first baseline:

| Partition | Source | Purpose |
| --- | --- | --- |
| Fitting | 167 manifest-selected images from `train/good` | Build the normal feature memory |
| Normal validation | 42 manifest-selected images from `train/good` | Analyze normal scores and select the threshold |
| Final test | Complete official `bottle/test` partition | Locked final detection and localization evaluation |
| Ground truth | `bottle/ground_truth` | Pixel-level localization evaluation |

The official test partition must not influence:

- feature-layer selection;
- backbone selection;
- image resolution;
- coreset ratio;
- distance metric;
- score normalization;
- anomaly-map postprocessing;
- threshold selection;
- model-selection decisions.

Official defect folder names may be used to group and explain final evaluation results. They must not be used as supervised fitting labels for the first anomaly model.

Synthetic validation anomalies are not required for the first implementation. If introduced later, they must be documented separately and must not be presented as equivalent to real test anomalies.

## Preprocessing Strategy

Preprocessing must be deterministic during fitting, validation, and final evaluation.

The selected preprocessing for the initial MVTec AD Bottle baseline is:

- decode the input image;
- convert it to RGB;
- resize it directly to 224 × 224 pixels;
- use bilinear interpolation with antialiasing;
- convert it to a `torch.float32` tensor;
- preserve channel-first ordering;
- normalize it with the ImageNet mean and standard deviation expected by the pretrained weights.

The resulting single-image tensor shape is:

```text
[1, 3, 224, 224]
```

This transformation has been executed successfully on normal and anomalous 900 × 900 RGB Bottle images.

The default TorchVision preprocessing pipeline, which first resizes to 256 × 256 pixels and then applies a 224 × 224 center crop, was inspected as an alternative. Direct resizing was selected because it preserves the complete bottle boundary and surrounding background, while center cropping removes part of the outer image area.

All 355 PNG files associated with the Bottle category are square at 900 × 900 pixels. Direct resizing therefore introduces no aspect-ratio distortion for this category. Later non-square categories must be evaluated separately and may require aspect-ratio-preserving resizing, padding, or category-specific preprocessing.

The inspection script retains both variants for comparison, but only direct 224 × 224 resizing belongs to the selected first-baseline preprocessing contract.

No random augmentation is used in the first baseline. If augmentation is investigated later, it must:

- be configured separately from deterministic evaluation preprocessing;
- preserve the definition of a normal sample;
- use a deterministic seed where randomness is involved;
- be evaluated against the unchanged baseline;
- never create apparent defects that are treated as ordinary normal data.

## Primary Model Strategy: PatchCore-Style Baseline

A PatchCore-style feature-memory pipeline is the selected first anomaly model because it:

- supports normal-only fitting;
- reuses pretrained feature representations;
- does not require full backbone retraining;
- supports image-level anomaly scoring;
- supports spatial anomaly localization;
- is compatible with CPU-oriented proof-of-concept development;
- makes the core anomaly-detection mechanism explicit and inspectable.

The intended fitting process is:

```text
Normal fitting image
→ deterministic preprocessing
→ frozen pretrained ResNet18
→ layer2 and layer3 feature maps
→ aligned multi-scale features
→ local patch embeddings
→ optional coreset selection
→ normal feature memory
```

The intended inference process is:

```text
Inspection image
→ identical preprocessing and embeddings
→ nearest normal-feature distances
→ patch anomaly scores
→ image anomaly score
→ spatial anomaly map
→ threshold decision
```

The first implementation should use the clearest correct mechanics before introducing advanced PatchCore optimizations.

## Reference Baseline Policy

A separate global-feature-distance method may later be implemented as a small image-level reference. It is not a prerequisite for the first PatchCore-style implementation.

A convolutional autoencoder is not part of the initial cycle because it introduces additional training, architecture, and reconstruction-loss decisions that do not directly validate the selected pretrained-feature strategy.

Any additional baseline must be evaluated with the same split isolation and experiment-recording rules.

## Backbone Strategy

The selected initial backbone is ResNet18 with default pretrained TorchVision weights.

The selected outputs are:

```text
layer2: [1, 128, 28, 28]
layer3: [1, 256, 14, 14]
```

After resizing and concatenation, one image currently produces:

```text
784 patch embeddings × 384 values
```

ResNet18 is fixed for the first baseline. A different backbone should be investigated only after:

- the complete baseline is correct and reproducible;
- score distributions and errors have been reviewed;
- runtime and feature-memory size have been measured;
- simpler causes of weak performance have been excluded.

Future backbone comparison must consider:

- validation evidence;
- CPU inference time;
- feature-map resolution;
- embedding dimensionality;
- feature-memory size;
- ONNX export compatibility;
- .NET runtime compatibility;
- pretrained-weight licensing and provenance.

## Feature Aggregation Strategy

The current verified aggregation performs:

1. extract `layer2` and `layer3`;
2. resize `layer3` to the 28 × 28 spatial resolution of `layer2` using bilinear interpolation;
3. concatenate both tensors along the channel dimension;
4. rearrange the combined tensor into one 384-dimensional embedding for each of 784 spatial positions.

This is the initial baseline aggregation. More advanced local pooling or PatchCore-specific aggregation may be compared later, but only through a separately recorded experiment.

## Coreset Strategy

The first feature-memory implementation should initially measure the complete fitting embedding collection before reduction.

With 167 fitting images and 784 embeddings per image, the unreduced collection is expected to contain:

```text
167 × 784 = 130,928 patch embeddings
```

At 384 `float32` values per embedding, the raw tensor contains approximately 50.3 million floating-point values before serialization overhead.

The implementation must measure the actual:

- extraction duration;
- tensor dimensions;
- in-memory size;
- serialized size;
- exact nearest-neighbor runtime.

A coreset should be introduced only if the full memory is impractical or if a controlled comparison demonstrates a useful size/runtime trade-off.

Open coreset decisions include:

- selection algorithm;
- sampling ratio;
- deterministic seed;
- batching strategy;
- quality and runtime comparison.

## Feature Memory Strategy

The feature memory must be treated as a model artifact tied to:

- dataset family and category;
- fitting split manifest;
- preprocessing configuration;
- backbone and pretrained weights;
- selected feature layers;
- feature aggregation;
- embedding dimensions;
- coreset configuration where applicable;
- framework versions.

Feature memories from incompatible configurations must never be mixed.

The serialization format remains open. The first format should prioritize correctness, explicit metadata, and reliable loading over compactness.

## Nearest-Neighbor Strategy

The first Python implementation should use exact nearest-neighbor distance where practical.

It must define and record:

- distance metric;
- query batching;
- reference-memory data type;
- score tensor shape;
- deterministic behavior;
- CPU execution time;
- peak memory use.

Approximate search or specialized indexing should be introduced only if exact search is too slow or memory-intensive for the selected category.

The later .NET implementation may use a different numerical library only if cross-runtime score parity remains within a documented tolerance.

## Anomaly Map Strategy

Patch anomaly scores must be mapped back to the spatial 28 × 28 patch grid and resized to the original image or evaluation-mask dimensions.

The implementation must define:

- patch-grid reconstruction;
- interpolation method;
- optional smoothing;
- score normalization for visualization;
- separation between raw scores and display colors;
- binary mask creation when a pixel threshold is applied.

Heatmap colors must never replace the underlying numerical anomaly values in stored results.

Any smoothing or normalization that affects metrics must be part of the recorded experiment configuration.

## Image-Level Scoring Strategy

The image-level anomaly score converts patch-level distances into one score per image.

Candidate methods include:

- maximum patch score;
- mean of the highest-scoring patches;
- a documented PatchCore-style reweighted score.

The first implementation should begin with the simplest clearly defined method. Alternative aggregation must be compared using validation evidence without tuning against final test labels.

## Threshold Strategy

The threshold converts a continuous anomaly score into a normal/anomalous decision.

The first threshold must be derived from the 42 normal validation images. The official Bottle test labels must not be used to select it.

Candidate initial methods include:

- a high percentile of normal validation scores;
- maximum normal validation score plus a documented margin;
- mean plus a documented number of standard deviations;
- a robust median and median-absolute-deviation rule.

Because the validation partition contains only normal images, it can constrain the false-positive behavior on known-normal data but cannot by itself optimize the trade-off against real anomalies.

The selected method, parameters, score distribution, threshold value, and limitations must be recorded.

Threshold-dependent test metrics may be reported after the threshold has been locked. They must not be used retroactively to change it within the same final experiment.

## Evaluation Strategy

The first final evaluation uses the complete official MVTec AD Bottle test partition.

### Image-Level Evaluation

Candidate metrics include:

- AUROC;
- Average Precision;
- Precision at the locked threshold;
- Recall at the locked threshold;
- F1 score at the locked threshold;
- confusion matrix;
- normal and anomalous score distributions.

### Pixel-Level Evaluation

Candidate metrics include:

- pixel-level AUROC;
- pixel-level Average Precision;
- pixel Precision, Recall, and F1 at a locked pixel threshold where defined;
- qualitative overlap between anomaly maps and supplied masks.

The exact required MVP metric set must be selected before final evaluation results are presented.

### Grouped Evaluation

Results may be grouped by the official Bottle test folders:

- `good`;
- `broken_large`;
- `broken_small`;
- `contamination`.

These groups support analysis and explanation. They remain excluded from normal-only fitting.

### Operational Evaluation

The following should also be measured:

- image decoding and preprocessing time;
- feature-extraction time;
- nearest-neighbor scoring time;
- total inference time;
- full feature-memory size;
- serialized artifact size;
- peak memory use where practical.

Performance values must state the hardware, software versions, input size, warm-up behavior, and measurement procedure.

## Qualitative Evaluation

Quantitative metrics must be supplemented with visual review of:

- correctly classified normal images;
- false positives;
- correctly detected anomalies;
- false negatives;
- accurately localized defects;
- diffuse or misplaced anomaly maps;
- very small defects whose visibility may be reduced by resizing;
- defects near the bottle boundary;
- the strongest and weakest examples per defect group.

Qualitative examples must be selected transparently and must not imply that a few favorable images represent complete model behavior.

Dataset license restrictions must be checked before any source image, mask, heatmap overlay, or derived visualization is committed publicly.

## Experiment Management

Every meaningful experiment must record at least:

- experiment identifier;
- execution timestamp;
- code revision where Git is available;
- Python and dependency versions;
- dataset and category;
- dataset archive checksum or dataset version reference;
- split-manifest path and checksum;
- random seed;
- backbone and pretrained weights;
- selected feature layers;
- input size and preprocessing;
- aggregation method;
- feature-memory or coreset configuration;
- distance metric;
- image-score aggregation;
- anomaly-map postprocessing;
- threshold method and value;
- evaluation metrics;
- runtime and memory measurements;
- output artifact locations;
- warnings and known limitations.

Human-readable summaries should be backed by machine-readable experiment configuration and result files.

## Reproducibility Strategy

Reproducibility requires:

- pinned direct dependencies;
- recorded Python version;
- deterministic file enumeration;
- versioned split manifests;
- explicit seeds for Python, PyTorch, NumPy, and any sampling code used;
- explicit preprocessing parameters;
- explicit model and pretrained-weight identifiers;
- isolated test data;
- versioned experiment configurations;
- recorded artifact checksums;
- no machine-specific path embedded in released artifacts.

The repository currently records Python 3.12.10, pinned direct dependencies, and the deterministic Bottle split manifest.

## Artifact Export Strategy

The final evaluated model package is expected to contain:

```text
feature-extractor.onnx
feature-memory artifact
model-metadata.json
preprocessing.json
thresholds.json
evaluation-summary.json
checksums.txt
license and attribution notices
```

These files must be exported together because their compatibility depends on shared preprocessing, feature layers, embedding dimensions, scoring behavior, and threshold configuration.

The existing ONNX export is provisional and must not be released as a complete anomaly model.

## Cross-Runtime Validation

Before the .NET backend consumes a released model package, Python and .NET must be compared using fixed fixtures.

Parity checks should include:

- decoded and preprocessed real images;
- normalized input tensors;
- ONNX feature outputs;
- combined patch embeddings;
- nearest-neighbor distances;
- patch anomaly scores;
- image-level scores;
- anomaly-map dimensions and representative values;
- threshold decisions.

Numeric tolerances must be documented. A visually plausible result is not sufficient evidence of parity.

## Local Hardware Strategy

The first development cycle targets CPU execution.

The current system has demonstrated approximately 12–13 milliseconds for a full pretrained ResNet18 execution on an artificial 224 × 224 input after warm-up. This is a technical reference only and excludes the remainder of the anomaly pipeline.

CPU feasibility must be measured again after preprocessing, feature-memory comparison, anomaly-map generation, and serialization are implemented.

GPU acceleration is optional and must not become necessary for reproducing the first baseline.

## Model Selection Gate

A candidate implementation may be approved as the first evaluated model only when:

- dataset licensing and attribution are documented;
- dataset validation passes;
- split isolation is verified;
- the experiment is reproducible;
- preprocessing is documented and visually reviewed;
- the validation strategy is documented;
- test data was not used for tuning;
- image-level metrics are recorded;
- localization metrics are recorded;
- qualitative errors are reviewed;
- CPU runtime is measured;
- feature-memory size is measured;
- ONNX export succeeds;
- ONNX numerical parity is within a defined tolerance;
- known limitations are documented.

Exact performance acceptance thresholds remain open until the first complete baseline provides measured evidence.

## Model Release Gate

A model artifact may be released publicly only when:

- redistribution of its weights and data-derived artifacts is permitted;
- all required notices and citations are included;
- no private dataset paths, images, credentials, or secrets are embedded;
- the artifact schema is versioned;
- integrity checks pass;
- evaluation metrics are linked to the artifact version;
- the Model Card is complete;
- intended and prohibited uses are documented;
- Python and .NET compatibility has been verified when the backend consumes the artifact.

## Failure Analysis Strategy

When performance is insufficient, investigation should proceed in this order:

1. Verify data integrity and split isolation.
2. Verify preprocessing and mask alignment.
3. Inspect normal and anomalous score distributions.
4. Review false positives and false negatives.
5. Verify feature-map and anomaly-map geometry.
6. Evaluate threshold behavior.
7. Measure feature-memory coverage.
8. Adjust scoring or feature aggregation.
9. Introduce or adjust coreset reduction.
10. Compare a different backbone only after earlier causes are understood.

This order reduces the risk of hiding pipeline errors through model complexity.

## Deferred Model Work

The following work is deferred from the first model-development cycle:

- simultaneous support for all dataset categories;
- MVTec LOCO AD model fitting;
- MVTec AD 2 model fitting and private-server evaluation;
- supervised defect-type classification;
- backbone fine-tuning;
- convolutional autoencoder development;
- real-time camera streams;
- continual learning;
- automated retraining;
- distributed or multi-GPU training;
- production deployment optimization;
- real pharmaceutical product images;
- regulatory validation;
- automated production acceptance or rejection decisions.

## Completed Preparation Steps

The following preparation steps are complete:

1. Establish the Python environment and pinned direct dependencies.
2. Verify pretrained ResNet18 CPU execution.
3. Verify `layer2` and `layer3` feature extraction.
4. Verify multi-scale patch-embedding construction.
5. Export and validate the provisional ONNX feature extractor.
6. Verify PyTorch/ONNX Runtime numerical parity.
7. Download, checksum, extract, inspect, and validate all three MVTec datasets.
8. Select MVTec AD Bottle for the first baseline.
9. Create and validate the deterministic 167/42 split with seed `42`.
10. Compare the default center-crop pipeline with direct 224 × 224 resizing on normal and anomalous Bottle images.
11. Add optional schema-versioned JSON reporting to all three dataset validators.
12. Validate MVTec LOCO AD masks against the category-specific pixel values defined in `defects_config.json`.
13. Select direct 224 × 224 resizing as the initial Bottle preprocessing contract.

## Immediate Next Steps

The next steps are:

1. Refactor the technical feature extractor into reusable, testable modules.
2. Implement deterministic loading from the Bottle split manifest.
3. Add automated tests for dataset report generation and schema contents.
4. Extract embeddings for the 167 fitting images.
5. Build the complete initial feature memory without coreset reduction.
6. Measure extraction duration, memory size, serialization size, and exact-search runtime.
7. Implement patch-level and image-level anomaly scoring.
8. Score the 42 normal validation images and lock the first threshold method.
9. Evaluate the unchanged baseline on the official Bottle test partition.
10. Generate metrics, anomaly maps, qualitative examples, and a reproducible experiment report.
11. Evaluate coreset reduction only after the complete-memory baseline is understood.

Only the first pending step should be started next so that each implementation decision remains understandable and verifiable.

## Related Documentation

- `DevelopmentStatus.md` records the verified implementation state and immediate progress.
- `ArchitectureOverview.md` describes the current and intended end-to-end architecture.
- `DatasetDocumentation.md` records dataset sources, licenses, structures, validation results, and selection evidence.
- `ProjectSpecification.md` defines the current functional and non-functional scope.
- A future `ModelCard.md` will document the selected evaluated model artifact.

## Last Updated

This strategy reflects the verified project state and selected first-baseline decisions as of 2026-08-13.
