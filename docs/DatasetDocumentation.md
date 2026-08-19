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

License reference: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

Project rules:

- archives, images, masks, and extracted subsets must not be committed to Git;
- commercial use requires resolving the non-commercial restriction with the owner;
- attribution and possible share-alike obligations must be preserved;
- screenshots and dataset-derived artifacts require separate redistribution review;
- uncertainty must be resolved using the complete license or directly with MVTec.

This document is not legal advice.

## Local Storage

```text
C:\dev\data\industrial-visual-anomaly-detection\
├── archives\
└── raw\
    ├── mvtec-ad\
    ├── mvtec-loco-ad\
    └── mvtec_ad_2\
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

Implemented validation establishes structure, inventory, PNG readability, recorded sizes and modes, image-to-mask relationships, compatible dimensions, and valid mask content.

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
- normal validation images determine the threshold;
- fitting and validation partitions must not overlap;
- manifests must cover their complete normal source set exactly once;
- generalized directory-based training creates a deterministic fitting and validation split from the discovered normal images;
- MVTec split-manifest paths remain relative to the configured dataset root;
- generalized `training_split.json` paths remain relative to the supplied normal-image directory;
- test images and masks must not populate feature memory or determine thresholds.

Bottle and Capsule test data was inspected and evaluated during development. Their current metrics are exploratory, not untouched final benchmark estimates.

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

This verification confirms that the generalized exporter can train and use a second category from a normal-image directory. Because the images still originate from MVTec AD, it does not yet demonstrate transfer to an independent real-world dataset.

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

## Future Dataset Evaluation

### MVTec LOCO AD

LOCO can test structural anomalies and logical violations. Local patch matching may be less suitable for anomalies requiring global context; this remains an untested hypothesis.

Before evaluation, define the validation strategy, multiple-mask handling, localization metrics, treatment of the seven-image discrepancy, and CPU/memory budget.

### MVTec AD 2

AD 2 can test lighting shifts and harder scenarios. Before evaluation, define handling of regular and shifted lighting variants, use of official validation, public versus private reporting, the evaluation-service workflow, and high-resolution resource controls.

## Dataset-Derived Artifacts

Feature memories are derived from normal images and may retain source information. They remain in ignored local output directories, are not committed or published before license review, and must record their dataset, category, configuration, and size.

Each artifact generated through the generalized directory workflow also contains `training_split.json`. This sidecar records the deterministic fitting and validation split using paths relative to the supplied image directory.

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
- MVTec AD 2: Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*, arXiv:2503.21622, 2025.

Exact author lists and publication details must be checked against the official source before release.

## Current Decision Status

| Decision | Status |
| --- | --- |
| Acquisition and extraction | Complete for all three families |
| Archive evidence | Sizes and SHA-256 values recorded |
| Local validation | Passed for all three families |
| Active family | MVTec AD |
| First baseline | Bottle, evaluated exploratorily and verified with the generalized exporter |
| Current artifact reference | Capsule 320 × 320, full memory, top-1%-mean aggregation |
| Bottle split | 167 fitting / 42 validation, seed 42 |
| Capsule split | 175 fitting / 44 validation, seed 42 |
| Pixel-level benchmark metrics | Planned |
| LOCO and AD 2 model evaluation | Planned |
| Screenshot redistribution | Open |
| Dataset-derived artifact redistribution | Open |
| Generalized fitting input | Recursive normal-image directory with PNG and JPEG files |
| Generalized split record | Artifact-local `training_split.json` |

## Next Dataset Tasks

1. Evaluate the generalized exporter with an independently sourced, user-controlled normal-image collection.
2. Define practical recommendations for minimum image count, image quality, and acceptable production variation.
3. Define an evaluation workflow for separate normal and anomalous image directories.
4. Select a third MVTec AD category beyond Bottle and Capsule.
5. Implement pixel-level localization metrics using existing masks.
6. Define a protocol that avoids further configuration selection on inspected test data.
7. Add duplicate-content detection if it provides meaningful split-safety evidence.
8. Define the LOCO evaluation protocol and treatment of its local count discrepancy.
9. Complete redistribution review before publishing dataset visuals or artifacts.
10. Remove absolute paths from any validation report selected for publication.

## Related Documentation

- `ProjectSpecification.md` – dataset requirements and acceptance criteria
- `ModelDevelopmentStrategy.md` – evaluation and model-selection methodology
- `ArchitectureOverview.md` – data and artifact boundaries
- `DevelopmentStatus.md` – verified implementation progress
- `ModelCard.md` – planned released-model evidence and limitations

## Last Updated

2026-08-19
