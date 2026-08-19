# Industrial Visual Anomaly Detection – Dataset Documentation

## Document Purpose

This document records dataset sources, licenses, local acquisition evidence, validation results, split definitions, selection decisions, and handling rules for the Industrial Visual Anomaly Detection project.

It distinguishes official source information, locally verified facts, project decisions, exploratory use, and open publication questions.

## Status Terms

- **Verified** – confirmed against a local archive or generated result.
- **Selected** – adopted for a current project reference.
- **Exploratory** – used during development without claiming an untouched final benchmark.
- **Planned** – intended but not yet implemented.
- **Open** – not yet decided or externally confirmed.

## Dataset Summary

| Dataset | Purpose | Officially described scale | Local status | Project role |
| --- | --- | ---: | --- | --- |
| MVTec AD | Industrial anomaly detection and localization | More than 5,000 images, 15 categories | Verified | Active model-development dataset |
| MVTec LOCO AD | Structural and logical anomaly detection | 3,644 stated inspection images, 5 categories | Verified with local count discrepancy | Later generalization candidate |
| MVTec AD 2 | Advanced anomaly detection under challenging conditions | More than 8,000 images, 8 scenarios | Verified | Later robustness candidate |
| VisA | Visual anomaly detection and localization across multiple domains | 10,821 images, 12 categories | Archive verified; Candle split exercised | Active generalized-workflow evaluation dataset |

The generalized training workflow also accepts user-provided normal-image directories. Such image collections are not treated as official benchmark datasets and require their own provenance, licensing, privacy, and redistribution records.

## Official Sources and Licenses

### MVTec AD

- Official page: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- Purpose: unsupervised anomaly detection and localization.
- Composition: 15 object and texture categories with normal training data, normal and anomalous test data, and pixel-precise annotations.
- License stated by MVTec: CC BY-NC-SA 4.0.
- Source review date: 2026-08-12.

### MVTec LOCO AD

- Official page: <https://www.mvtec.com/company/research/datasets/mvtec-loco>
- Purpose: anomaly detection involving structural defects and violations of logical constraints.
- Official composition: 3,644 inspection images across 5 categories.
- License stated by MVTec: CC BY-NC-SA 4.0.
- Source review date: 2026-08-12.

### MVTec AD 2

- Official page: <https://www.mvtec.com/company/research/datasets/mvtec-ad-2>
- Purpose: advanced unsupervised anomaly detection with challenging acquisition conditions and distribution shifts.
- Composition: 8 scenarios with training, validation, public test, private test, and mixed private test data.
- License stated by MVTec: CC BY-NC-SA 4.0.
- Source review date: 2026-08-12.

### VisA

- Official registry: <https://registry.opendata.aws/visa/>
- Project documentation: <https://github.com/amazon-research/spot-diff>
- Purpose: visual anomaly detection and segmentation.
- Official composition: 10,821 images across 12 categories, including 9,621 normal and 1,200 anomalous images.
- Annotations: image-level labels and pixel-level anomaly masks.
- License stated by the official registry: CC BY 4.0.
- Source review date: 2026-08-19.

License references:

- MVTec datasets: <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- VisA: <https://creativecommons.org/licenses/by/4.0/>

Project rules:

- archives, images, masks, and extracted subsets must not be committed to Git;
- commercial use requires resolving the non-commercial restriction with the owner;
- attribution and possible share-alike obligations must be preserved;
- screenshots and dataset-derived artifacts require separate redistribution review;
- uncertainty must be resolved using the applicable complete license or directly with the respective dataset owner.

This document is not legal advice.

## Local Storage

```text
C:\dev\data\industrial-visual-anomaly-detection\
├── archives\
└── raw\
    ├── mvtec-ad\
    ├── mvtec-loco-ad\
    ├── mvtec_ad_2\
    └── visa\
```

These are machine-specific locations, not repository defaults. Code receives dataset roots through arguments or configuration.

## Local Acquisition and Validation

### MVTec AD

- Acquisition date: 2026-08-12.
- Archive: `mvtec_anomaly_detection.tar.xz`.
- Size: 5,264,982,680 bytes.
- SHA-256: `CF4313B13603BEC67ABB49CA959488F7EEDCE2A9F7795EC54446C649AC98CD3D`.
- Download, extraction, and structural validation: complete and passed.
- Official checksum comparison: not performed.
- Validator: `scripts/validate_mvtec_ad.py`.

Verified inventory:

- 15 categories;
- 3,629 normal training images;
- 467 normal test images;
- 1,258 anomalous test images;
- 5,354 inspection images in total;
- 1,258 ground-truth masks;
- 6,612 readable PNG files including masks;
- 4,141 RGB and 2,471 grayscale files;
- sizes: 700², 800², 840², 900², 1,000², and 1,024² pixels.

All anomalous test images have matching readable binary single-channel masks with identical dimensions and positive anomaly pixels.

Categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and `zipper`.

### MVTec LOCO AD

- Acquisition date: 2026-08-12.
- Archive: `mvtec_loco_anomaly_detection.tar.xz`.
- Size: 6,126,996,224 bytes.
- SHA-256: `9E7C84DBA550FD2E59D8E9E231C929C45BA737B6B6A6D3814100F54D63AAE687`.
- Download, extraction, and structural validation: complete and passed.
- Official checksum comparison: not performed.
- Validator: `scripts/validate_mvtec_loco_ad.py`.

Verified inventory:

- 5 categories;
- 1,778 normal training images;
- 305 normal validation images;
- 575 normal test images;
- 561 logical and 432 structural anomaly test images;
- 993 anomalous test images and mask groups;
- 3,651 inspection images in total;
- 1,246 mask files;
- 4,897 readable PNG files including masks;
- 3,651 RGB and 1,246 grayscale files;
- five sizes ranging from 800 × 1,600 to 1,700 × 1,000 pixels.

All anomaly images have matching non-empty mask groups. All mask files are readable, single-channel, geometrically compatible, and contain a configured positive anomaly value. Positive values are category-specific and validated against `defects_config.json`; LOCO masks are not necessarily binary 0/255 masks.

Categories: `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, and `splicing_connectors`.

#### Local Count Discrepancy

The archive contains 3,651 inspection images while the official page states 3,644. The difference is isolated to `splicing_connectors`:

- 360 local normal training images versus 354 in the published table;
- 60 local normal validation images versus 59 in the published table;
- additional training names `354.png` through `359.png`;
- additional validation name `059.png`.

The seven files remain part of the acquired archive. Their origin is not officially confirmed. Future LOCO experiments must identify this archive by checksum and record an exact split.

### MVTec AD 2

- Acquisition date: 2026-08-12.
- Archive: `mvtec_ad_2.tar.gz`.
- Size: 32,739,596,982 bytes.
- SHA-256: `C0DED99EF32BFC8E352D52BEB44515E5B292B8598CB963AADFA91CA0763505E4`.
- Download, extraction, and structural validation: complete and passed.
- Official checksum comparison: not performed.
- Validator: `scripts/validate_mvtec_ad_2.py`.

Verified inventory:

- 8 scenarios;
- 2,528 normal training images;
- 302 normal validation images;
- 379 public normal and 705 public anomalous test images;
- 705 public masks;
- 2,045 private and 2,045 mixed private test images;
- 8,004 inspection images in total;
- 8,709 readable PNG files including masks;
- 5,486 RGB and 3,223 grayscale files;
- five sizes ranging from 1,400 × 1,900 to 4,224 × 1,056 pixels.

All public anomalous images have matching readable binary masks with identical dimensions and positive pixels. Private ground truth is unavailable locally; complete private evaluation requires the official MVTec process.

Scenarios: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, and `walnuts`.

### VisA

- Acquisition date: 2026-08-19.
- Archive: `VisA_20220922.tar`.
- Verified size: 1,929,840,640 bytes.
- SHA-256: `2EB8690C803AB37DE0324772964100169EC8BA1FA3F7E94291C9CA673F40F362`.
- Source: `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`.
- Download and extraction: complete.
- Official checksum comparison: not available through the reviewed registry entry.
- Included dataset license file: `LICENSE-DATASET`.

Extracted categories:

`candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, `macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, and `pipe_fryum`.

Verified Candle inventory:

- 1,000 normal JPEG images;
- 100 anomalous JPEG images;
- 100 anomaly-mask images;
- official one-class split with 900 normal training images, 100 normal test images, and 100 anomalous test images;
- every Candle image referenced by the reviewed split CSV exists locally.

The official one-class split is defined by `split_csv/1cls.csv`. The category-local `image_anno.csv` provides image labels and mask relationships but does not define the train/test partition.

## Machine-Readable Validation Reports

All validators support `--report` and write schema-versioned UTF-8 JSON after all implemented checks pass. Schema version `1` records dataset identity, resolved root, status, categories, inventories, image dimensions and modes, and mask-validation counts.

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_mvtec_ad.py `
    --dataset-root C:\path\to\mvtec-ad `
    --report .\validation-reports\mvtec-ad.json
```

Reports are ignored under `validation-reports/`. They contain local absolute paths and must be sanitized before publication.

## Repository and Handling Policy

The repository may contain source code, tests, documentation, deterministic split manifests, and non-sensitive configuration.

It shall not contain archives, extracted images or masks, dataset subsets, local validation reports, feature memories, model artifacts, or temporary visualization output.

Raw dataset files are immutable inputs. Project scripts shall not alter, rename, move, or delete them.

## Meaning of Dataset Validation

Implemented MVTec validation establishes structure, inventory, PNG readability, recorded sizes and modes, image-to-mask relationships, compatible dimensions, and valid mask content. The current VisA verification establishes archive identity, extraction structure, Candle inventory, official split counts, referenced-image existence, and successful model ingestion; a dedicated full VisA validator is not yet implemented.

It does not prove annotation semantics, model performance, archive equality with an official checksum, redistribution rights, unbiased evaluation, or production suitability.

## External Normal-Image Input Contract

The generalized training workflow accepts a directory containing normal images for one coherent object or texture category.

Input requirements:

- supported formats are PNG, JPEG, and JPG, matched case-insensitively;
- images are discovered recursively;
- the directory must contain at least two unique image paths;
- all supplied images must represent normal, defect-free examples;
- fitting data should represent the expected production variation in position, lighting, background, and appearance;
- images from unrelated categories must not be mixed into one artifact;
- anomalous evaluation images must remain outside the supplied normal-image directory.

The exporter creates deterministic fitting and validation partitions using the configured validation fraction and random seed. It records the resulting relative paths and counts in `training_split.json` inside the artifact directory.

The split manifest supports reproducibility but does not establish image provenance, licensing, data quality, or production suitability.

## Split and Test-Isolation Policy

- normal fitting images construct feature memory;
- normal validation images determine the initial threshold through the configured normal-score quantile;
- fitting and validation partitions must not overlap;
- manifests must cover their complete normal source set exactly once;
- generalized directory-based training creates a deterministic fitting and validation split from the discovered normal images;
- MVTec split-manifest paths remain relative to the configured dataset root;
- generalized `training_split.json` paths remain relative to the supplied normal-image directory;
- test images and masks must not populate feature memory or determine the initial threshold;
- comparing threshold variants after inspecting labeled test results is exploratory calibration and invalidates those same results as an untouched final estimate.

Bottle and Capsule test data was inspected and evaluated during development. Their current metrics are exploratory, not untouched final benchmark estimates.

The official VisA Candle test split was used to compare q100, q99, and q95 thresholds. The q95 result is therefore a provisional calibration result, not an independently validated final estimate. Further quantile tuning against the same 200 Candle test images is intentionally excluded.

Future unbiased selection requires a new untouched holdout, an applicable official validation protocol, controlled synthetic development anomalies identified as synthetic, or another documented convention.

## Active MVTec AD Categories

### Bottle Baseline

Bottle was selected first because its object position, background, and lighting are largely consistent; defects are understandable; masks are available; and direct resizing preserves the complete circular boundary.

Split:

- manifest: `configs/splits/mvtec-ad-bottle-seed-42.json`;
- seed: 42;
- source: 209 images from `bottle/train/good`;
- fitting: 167;
- validation: 42;
- overlap: 0;
- complete source coverage.

Reference preprocessing and use:

- direct 224 × 224 bilinear resize with antialiasing;
- RGB conversion, tensor conversion, and ImageNet normalization;
- complete normal fitting memory;
- official Bottle test groups used for exploratory evaluation;
- group names used only for reporting;
- masks excluded from fitting and threshold selection.

Bottle test data influenced aggregation analysis, so its results remain exploratory.

Generalized exporter verification:

- input directory: `bottle/train/good`;
- supported directory-based discovery used instead of a pre-existing split manifest;
- source images: 209;
- fitting images: 167;
- validation images: 42;
- split seed: 42;
- validation fraction: 0.2;
- input size: 320 × 320;
- feature memory: 267,200 × 384;
- feature-memory size: approximately 391.41 MiB;
- threshold: approximately `3.216314`;
- normal Bottle test image classified as normal;
- `broken_large` Bottle test image classified as anomalous;
- generated split recorded in `training_split.json`.

This verification confirms that the generalized exporter can train and use a second category from a normal-image directory. Because the images still originate from MVTec AD, this Bottle check alone does not demonstrate transfer to an independent dataset. The later VisA Candle experiment provides that separate dataset exercise.

### Capsule Reference

Capsule was selected second to test transfer of the generic pipeline to another object geometry. Test groups are `crack`, `faulty_imprint`, `good`, `poke`, `scratch`, and `squeeze`.

Split:

- manifest: `configs/splits/mvtec-ad-capsule-seed-42.json`;
- seed: 42;
- source: 219 images from `capsule/train/good`;
- fitting: 175;
- validation: 44;
- overlap: 0;
- complete source coverage.

Selected reference configuration:

- 320 × 320 input;
- 40 × 40 patch grid;
- 280,000 × 384 complete feature memory;
- approximately 410.16 MiB feature-memory size;
- top-1%-patch mean aggregation;
- maximum normal validation score as threshold;
- threshold approximately `2.501822`;
- memory fraction 1.0.

Capsule test results are exploratory. Its reference model was exported locally as a Python/PyTorch artifact, which is excluded from Git and not approved for redistribution.

## Active VisA Category

### Candle Generalized Workflow

Candle was selected as the first category from a dataset family outside MVTec. The official one-class split supplies 900 normal training images and a balanced test partition containing 100 normal and 100 anomalous images.

Prepared fitting input:

```text
prepared/candle/train/normal
```

The 900 official normal training images were copied into the prepared directory without modifying the extracted source files. The generalized exporter then created a deterministic internal split:

- source images: 900;
- fitting images: 720;
- validation images: 180;
- validation fraction: 0.2;
- split seed: 42;
- input size: 320 × 320;
- feature memory: 288,000 × 384;
- feature-memory sampling fraction: 0.25;
- sampling seed: 42;
- top-score fraction: 0.01.

Dataset-independent evaluation manifest:

```text
prepared/candle/evaluation_manifest.csv
```

The manifest contains 100 official normal test images and 100 official anomalous test images with `image`, `group`, and `is_anomalous` columns. Paths are relative to the extracted VisA dataset root.

Threshold comparison:

| Variant | Quantile | Threshold | TP | TN | FP | FN | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q100 | 1.00 | 3.373678 | 31 | 100 | 0 | 69 | 1.0000 | 0.3100 | 1.0000 | 0.4733 |
| q99 | 0.99 | 3.001366 | 56 | 100 | 0 | 44 | 1.0000 | 0.5600 | 1.0000 | 0.7179 |
| q95 | 0.95 | 2.763051 | 69 | 95 | 5 | 31 | 0.9324 | 0.6900 | 0.9500 | 0.7931 |

All three variants use feature-memory files with identical SHA-256 hashes. Their scores are therefore directly comparable, and their classification differences result from threshold selection.

q95 is the provisional calibration candidate for a review-oriented workflow. Because the official test split was inspected while selecting it, future validation must keep q95 fixed and use previously unused evidence.

## Future Dataset Evaluation

### MVTec LOCO AD

LOCO can test structural anomalies and logical violations. Local patch matching may be less suitable for anomalies requiring global context; this remains an untested hypothesis.

Before evaluation, define the validation strategy, multiple-mask handling, localization metrics, treatment of the seven-image discrepancy, and CPU/memory budget.

### MVTec AD 2

AD 2 can test lighting shifts and harder scenarios. Before evaluation, define handling of regular and shifted lighting variants, use of official validation, public versus private reporting, the evaluation-service workflow, and high-resolution resource controls.

## Dataset-Derived Artifacts

Feature memories are derived from normal images and may retain source information. They remain in ignored local output directories, are not committed or published before license review, and must record their dataset, category, configuration, and size.

Each artifact generated through the generalized directory workflow also contains `training_split.json`. This sidecar records the deterministic fitting and validation split using paths relative to the supplied image directory.

Schema-version-2 artifact metadata records the threshold method and quantile. Loading remains compatible with schema-version-1 artifacts through maximum-normal defaults.

The split manifest supports reproducibility but does not grant permission to redistribute the source images or the resulting feature memory.

## Publication Policy

Until redistribution review is complete:

- do not publish original images, masks, screenshots, archives, or subsets;
- do not publish feature memories or packaged model artifacts;
- prefer diagrams, artificial tensors, self-created images, or clearly permitted assets;
- link to official dataset pages and report aggregate facts with attribution.

## Citation Records

Public work shall preserve the citation requested by each official dataset source. The local readme files identify:

- MVTec AD: Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*;
- MVTec LOCO AD: Bergmann et al., *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization*, IJCV, 2022;
- MVTec AD 2: Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*, arXiv:2503.21622, 2025;
- VisA: Zou, Jeong, Pemula, Zhang, and Dabeer, *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, arXiv:2207.14315, 2022.

Exact author lists and publication details must be checked against the official source before release.

## Current Decision Status

| Decision | Status |
| --- | --- |
| Acquisition and extraction | Complete for MVTec AD, MVTec LOCO AD, MVTec AD 2, and VisA |
| Archive evidence | Sizes and SHA-256 values recorded |
| Local validation | Passed for all three MVTec families; VisA archive and Candle split verified |
| Active families | MVTec AD and VisA |
| First baseline | Bottle, evaluated exploratorily and verified with the generalized exporter |
| Established service artifact reference | Capsule 320 × 320, full memory, top-1%-mean aggregation |
| Current calibration candidate | VisA Candle q95, pending independent confirmation |
| Bottle split | 167 fitting / 42 validation, seed 42 |
| Capsule split | 175 fitting / 44 validation, seed 42 |
| Pixel-level benchmark metrics | Planned |
| LOCO and AD 2 model evaluation | Planned |
| Screenshot redistribution | Open |
| Dataset-derived artifact redistribution | Open |
| Generalized fitting input | Recursive normal-image directory with PNG and JPEG files |
| Generalized split record | Artifact-local `training_split.json` |
| Dataset-independent evaluation input | CSV manifest with `image`, `group`, and `is_anomalous` |
| Threshold metadata | Schema version 2 with method and quantile; schema version 1 remains loadable |

## Next Dataset Tasks

1. Keep q95 fixed and validate the threshold strategy on previously unused data or another suitable category.
2. Define a strict calibration and independent final-test protocol for future categories.
3. Define practical recommendations for minimum image count, image quality, and acceptable production variation.
4. Add a dedicated full VisA structural and mask validator if broader VisA use is pursued.
5. Select a third MVTec AD category beyond Bottle and Capsule.
6. Implement pixel-level localization metrics using existing masks.
7. Add duplicate-content detection if it provides meaningful split-safety evidence.
8. Define the LOCO evaluation protocol and treatment of its local count discrepancy.
9. Complete redistribution review before publishing dataset visuals or artifacts.
10. Remove absolute paths from any validation report selected for publication.

## Related Documentation

- `ProjectSpecification.md` – dataset requirements and acceptance criteria
- `ModelDevelopmentStrategy.md` – evaluation and model-selection methodology
- `ArchitectureOverview.md` – data and artifact boundaries
- `DevelopmentStatus.md` – verified implementation progress
- `experiments/visa-candle-threshold-calibration.md` – exploratory threshold comparison and methodological limitation
- `ModelCard.md` – planned released-model evidence and limitations

## Last Updated

2026-08-19