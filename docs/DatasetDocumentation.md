# Industrial Visual Anomaly Detection – Dataset Documentation

## Document Purpose

This document records the provenance, licensing, intended use, local handling, structural validation, and selection status of datasets considered for the Industrial Visual Anomaly Detection project.

It is a living document. Facts obtained from official sources are separated from observations that still require verification against the downloaded archives. MVTec AD and its `bottle` category have been selected for the first MVP model-development cycle.

## Documentation Status

The following status labels are used:

- **Officially documented** – supported by an official dataset page or publication;
- **Locally verified** – confirmed against the downloaded files by a repeatable inventory or validation process;
- **Open** – not yet inspected or decided;
- **Selected** – approved for the first model-development cycle through a documented decision.

Dataset acquisition and local technical validation are complete for MVTec AD, MVTec LOCO AD, and MVTec AD 2. Archive checksums, extracted structures, inventories, image readability, and available mask relationships have been verified locally. Dataset selection, experiment suitability, and model performance remain separate evaluation concerns.

## Candidate Dataset Summary

| Dataset | Official purpose | Official scale | Anomaly coverage | Current status |
| --- | --- | --- | --- | --- |
| MVTec AD | Industrial anomaly-detection benchmark | More than 5,000 images across 15 object and texture categories | Visual defects with pixel-precise annotations | Locally validated / `bottle` selected for MVP |
| MVTec LOCO AD | Unsupervised anomaly detection and localization with logical constraints | 3,644 images across 5 categories | Structural and logical anomalies with pixel-precise ground truth | Locally validated / later evaluation candidate |
| MVTec AD 2 | Advanced unsupervised industrial anomaly-detection benchmark | More than 8,000 images across 8 scenarios | Challenging conditions, public and private test portions, and lighting variation | Locally validated / later evaluation candidate |

The scale and purpose statements above come from the official MVTec dataset pages. They must not be treated as a substitute for a local inventory of the downloaded version.

## Official Sources

### MVTec AD

- Official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- Officially documented purpose: benchmarking anomaly-detection methods for industrial inspection.
- Officially documented composition: more than 5,000 high-resolution images in 15 object and texture categories.
- Officially documented learning setup: defect-free training images and test images containing normal and defective samples.
- Officially documented annotations: pixel-precise anomaly annotations.
- Officially stated license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- Local verification status: **Passed**.
- Source review date: 2026-08-12.

### MVTec LOCO AD

- Official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad>
- Officially documented purpose: evaluating unsupervised anomaly-detection and localization algorithms under logical constraints.
- Officially documented composition: 3,644 images in 5 categories.
- Officially documented anomaly types: structural anomalies and logical anomalies.
- Officially documented annotations: pixel-precise ground-truth data for anomalous regions.
- Officially stated license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- Local verification status: **Passed**.
- Source review date: 2026-08-12.

### MVTec AD 2

- Official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>
- Officially documented purpose: benchmarking unsupervised anomaly detection on advanced industrial inspection scenarios.
- Officially documented composition: more than 8,000 high-resolution images across 8 scenarios.
- Officially documented learning setup: defect-free training and validation images.
- Officially documented test setup: a public test portion with available anomaly annotations and a private portion whose ground truth is evaluated through the official evaluation server.
- Officially documented challenge: normal and anomalous test images may be captured under lighting conditions not represented in training data.
- Officially stated license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- Official evaluation server: <https://benchmark.mvtec.com/>
- Local verification status: **Passed**.
- Source review date: 2026-08-12.

## License And Usage Rules

All three candidate datasets are currently described by MVTec as licensed under CC BY-NC-SA 4.0.

Official license reference:

<https://creativecommons.org/licenses/by-nc-sa/4.0/>

The project must therefore treat the following as mandatory constraints:

- dataset use must remain compatible with the non-commercial restriction;
- required attribution must be preserved;
- applicable citation requirements must be documented;
- redistribution of original images must not be assumed to be permitted merely because the source archive was downloadable;
- share-alike obligations must be reviewed before distributing adapted dataset material;
- dataset-derived model artifacts must undergo a separate redistribution review before public release;
- uncertainty about an intended use must be resolved against the full license text or directly with the dataset owner.

This document records engineering constraints and does not constitute legal advice.

## Local Acquisition Records

### MVTec AD

- Acquisition date: 2026-08-12.
- Archive filename: `mvtec_anomaly_detection.tar.xz`.
- Archive size: 5,264,982,680 bytes.
- Locally calculated SHA-256: `CF4313B13603BEC67ABB49CA959488F7EEDCE2A9F7795EC54446C649AC98CD3D`.
- Download status: Complete.
- Extraction status: Complete.
- Structural validation status: Passed.
- Checksum status: Recorded locally; no official reference checksum has been compared.
- Categories: 15.
- Normal training images: 3,629.
- Normal test images: 467.
- Anomalous test images: 1,258.
- Ground-truth masks: 1,258.
- Total inspection images: 5,354.
- Mask consistency: All 1,258 anomalous test images have a matching, readable, binary single-channel mask with identical dimensions and at least one anomaly pixel.
- Readable PNG files: 6,612.
- Distinct image sizes: 6.
- Image sizes: 700 × 700, 800 × 800, 840 × 840, 900 × 900, 1,000 × 1,000, and 1,024 × 1,024 pixels.
- Image modes: 4,141 RGB files and 2,471 single-channel grayscale files.
- Validation script: `scripts/validate_mvtec_ad.py`.

### MVTec LOCO AD

- Acquisition date: 2026-08-12.
- Archive filename: `mvtec_loco_anomaly_detection.tar.xz`.
- Archive size: 6,126,996,224 bytes.
- Locally calculated SHA-256: `9E7C84DBA550FD2E59D8E9E231C929C45BA737B6B6A6D3814100F54D63AAE687`.
- Download status: Complete.
- Extraction status: Complete.
- Structural validation status: Passed.
- Checksum status: Recorded locally; no official reference checksum has been compared.
- Categories: 5.
- Normal training images: 1,778.
- Normal validation images: 305.
- Normal test images: 575.
- Logical anomaly test images: 561.
- Structural anomaly test images: 432.
- Total anomalous test images: 993.
- Total inspection images: 3,651.
- Ground-truth groups: 993.
- Ground-truth mask files: 1,246.
- Readable PNG files: 4,897.
- Distinct image sizes: 5.
- Image sizes: 800 × 1,600, 1,600 × 1,100, 1,600 × 1,280, 1,700 × 850, and 1,700 × 1,000 pixels.
- Image modes: 3,651 RGB files and 1,246 single-channel grayscale files.
- Mask consistency: All 993 anomalous test images have matching non-empty mask groups; all 1,246 masks are readable, binary, single-channel, geometrically compatible, and contain anomaly pixels.
- Validation script: `scripts/validate_mvtec_loco_ad.py`.
- Published-count discrepancy: The local archive contains seven additional normal `splicing_connectors` images compared with the statistical table cited in the dataset publication; the reason remains unconfirmed.

### MVTec AD 2

- Acquisition date: 2026-08-12.
- Archive filename: `mvtec_ad_2.tar.gz`.
- Archive size: 32,739,596,982 bytes.
- Locally calculated SHA-256: `C0DED99EF32BFC8E352D52BEB44515E5B292B8598CB963AADFA91CA0763505E4`.
- Download status: Complete.
- Extraction status: Complete.
- Structural validation status: Passed.
- Checksum status: Recorded locally; no official reference checksum has been compared.
- Scenarios: 8.
- Normal training images: 2,528.
- Normal validation images: 302.
- Public normal test images: 379.
- Public anomalous test images: 705.
- Public ground-truth masks: 705.
- Private test images: 2,045.
- Private mixed test images: 2,045.
- Total inspection images: 8,004.
- Readable PNG files including masks: 8,709.
- Distinct image sizes: 5.
- Image sizes: 1,400 × 1,900, 2,100 × 1,520, 2,232 × 1,024, 2,448 × 2,048, and 4,224 × 1,056 pixels.
- Image modes: 5,486 RGB files and 3,223 single-channel grayscale files.
- Mask consistency: All 705 public anomalous test images have matching, readable, binary single-channel masks with identical dimensions and at least one anomaly pixel.
- Private evaluation limitation: Ground truth for private test data is not locally available; complete official evaluation requires the MVTec evaluation server.
- Validation script: `scripts/validate_mvtec_ad_2.py`.

## Repository And Local Storage Policy

Original datasets must not be committed to the source repository.

The application must use configurable dataset roots. Documentation and configuration examples must use repository-relative placeholders or environment variables rather than developer-specific absolute paths.

A recommended local layout is:

```text
data/
├── raw/
│   ├── mvtec-ad/
│   ├── mvtec-loco-ad/
│   └── mvtec-ad-2/
├── derived/
│   ├── inventories/
│   ├── split-manifests/
│   └── validation-reports/
└── README.md
```

The `data/raw/` and large generated-data directories must be excluded through `.gitignore`. Small machine-readable inventories, split manifests, and validation reports may be versioned when they contain no restricted content, private paths, or prohibited derived data.

Downloaded archives may be stored outside the repository and referenced through configuration. Moving large archives into the repository directory is not required.

## Data Handling Principles

The project must:

1. preserve original archives or extracted source data as read-only inputs where practical;
2. avoid modifying the original dataset structure in place;
3. write generated manifests, caches, resized images, embeddings, and reports to separate derived-data locations;
4. record the source dataset family, version or acquisition date, archive filename, and checksum when available;
5. avoid storing developer-specific absolute paths in versioned reports;
6. prevent training, validation, and test samples from crossing split boundaries;
7. preserve the association between anomalous images and their ground-truth masks;
8. treat image files and archives as untrusted input;
9. document every transformation that creates derived image data;
10. verify redistribution rights before publishing images, masks, embeddings, Memory Banks, or other dataset-derived artifacts.

## Required Local Inventory

After extraction, the inventory process should record at least:

- dataset identifier;
- acquisition date;
- official source URL;
- downloaded archive filename;
- archive size;
- cryptographic checksum calculated locally;
- extracted root structure;
- category names;
- partition names;
- defect-type names;
- image extensions;
- image counts by category, partition, and label;
- image dimensions and channel modes;
- mask counts and dimensions;
- unreadable or unsupported files;
- empty directories;
- duplicate file hashes where measured;
- total extracted size;
- inventory-tool version or source revision.

The local checksum proves which downloaded file was inspected. Unless MVTec publishes a reference checksum, it does not independently prove that the archive is an official unmodified release.

## Structural Validation Requirements

Before a dataset may be used for fitting or evaluation, automated validation must confirm:

- the configured root exists;
- the selected category exists;
- required partitions exist;
- at least one normal training image exists;
- file enumeration is deterministic;
- every selected image can be decoded;
- every selected image has a supported color or channel representation;
- labels can be derived unambiguously from the documented structure;
- anomaly masks exist when required by the selected evaluation protocol;
- masks can be associated unambiguously with source images;
- mask geometry matches the corresponding image geometry or a documented mapping;
- no selected source file occurs in more than one project-defined split;
- no generated variant of one source image crosses split boundaries;
- validation failures produce a machine-readable report and prevent affected workflows from continuing silently.

Dataset-family-specific validators may be required because MVTec AD, MVTec LOCO AD, and MVTec AD 2 do not share one guaranteed directory contract.

## Split And Test-Isolation Policy

The official final test data must not be used to choose:

- the backbone;
- feature layers;
- preprocessing parameters;
- input resolution;
- augmentation settings;
- PatchCore aggregation behavior;
- Coreset settings;
- nearest-neighbor configuration;
- image-score calculation;
- score normalization;
- classification threshold;
- model-selection decisions.

A separate validation strategy must be established before model comparison begins.

Depending on the selected dataset family, this may use:

- an official validation partition;
- a deterministic holdout from normal training images;
- controlled synthetic validation anomalies;
- another published and explicitly documented convention.

Synthetic anomalies must be identified as synthetic. They must not be reported as equivalent to the real anomalies in a final benchmark test set.

The final test evaluation must be performed only after the selected configuration and decision threshold are locked. Subsequent tuning based on final test results requires a new clearly identified development cycle and must not be presented as an untouched final evaluation.

## Dataset-Family Considerations

### MVTec AD

Potential strengths for the first MVP:

- established anomaly-detection benchmark;
- several visually understandable object and texture categories;
- normal-only training structure;
- public pixel-precise annotations;
- practical starting point for a PatchCore-style implementation.

Items requiring local inspection:

- exact extracted directory structure;
- category-level image counts and dimensions;
- category-specific preprocessing risks;
- suitability of candidate categories for CPU-oriented experiments;
- validation strategy without final-test tuning;
- permitted use of screenshots in public documentation.

### MVTec LOCO AD

Potential strengths:

- separates structural defects from violations of logical constraints;
- supports a stronger industrial workflow narrative;
- provides a later test of whether local patch-based methods capture global relationships.

Potential limitation for the first MVP:

- a minimal PatchCore-style model may be more naturally suited to local structural anomalies than to logical anomalies that require broader context.

This is a technical hypothesis to test, not a reported result.

Items requiring local inspection:

- category structures and labels;
- representation of structural and logical anomaly types;
- ground-truth organization;
- appropriate metrics and official evaluation tooling;
- validation strategy;
- CPU cost at the original resolutions.

#### Local Count Discrepancy

The locally acquired archive contains 3,651 inspection images, while the official dataset page states 3,644 images.

The difference is isolated to the `splicing_connectors` category:

- the published statistical table reports 354 normal training images; the local archive contains 360;
- the published statistical table reports 59 normal validation images; the local archive contains 60;
- the local additional files are consecutively named `354.png` through `359.png` for training and `059.png` for validation;
- all other locally counted category and partition totals agree with the published statistical table.

The seven additional files are retained as part of the acquired archive. The reason for the difference has not been officially confirmed. Experiments must identify the local archive checksum and use a generated split manifest so that the exact evaluated dataset state remains reproducible.

### MVTec AD 2

Potential strengths:

- advanced and challenging inspection scenarios;
- official normal validation data;
- explicit distribution shifts involving lighting conditions;
- official utilities and evaluation infrastructure.

Potential limitations for the first MVP:

- higher complexity than the original MVTec AD benchmark;
- private ground truth for part of the test data;
- reliance on the official evaluation server for complete official evaluation;
- potentially greater CPU, memory, storage, and preprocessing demands.

Items requiring local inspection:

- archive and directory organization;
- separation of training, validation, public test, and private evaluation inputs;
- category dimensions and storage requirements;
- official submission format and evaluation constraints;
- treatment of lighting-condition subsets;
- suitability for the first CPU-oriented proof of concept.

## MVP Dataset Selection Criteria

The first selected category should:

- have a clearly defined normal state;
- contain enough normal fitting images for a meaningful Memory Bank;
- include understandable anomalies for qualitative review;
- provide usable reference labels and masks;
- support honest validation without using final test labels for tuning;
- fit within available CPU, memory, and storage constraints;
- produce useful localization examples;
- avoid unnecessary preprocessing ambiguity;
- support a coherent industrial portfolio narrative;
- allow compliant public documentation.

Candidate MVTec AD categories currently mentioned for inspection include:

- `bottle`;
- `capsule`;
- `pill`.

The shortlist was used for the initial manual comparison. The `bottle` category has since been selected for the first MVP model-development cycle; `capsule` and `pill` remain possible later evaluation categories.

### Preliminary MVTec AD Bottle Review

The `bottle` category has passed an initial manual suitability review.

Observed characteristics:

- object position is largely consistent;
- background appearance is largely consistent;
- lighting variation appears limited;
- `broken_large`, `broken_small`, and `contamination` anomalies are visually understandable;
- the inspected ground-truth mask aligns plausibly with the visible defect;
- normal and anomalous samples appear suitable for an explainable first anomaly-detection experiment.

The `bottle` category is selected for the first MVP model-development cycle.

This review is qualitative and based on a limited visual sample. It does not constitute model evaluation or prove robustness. The category must still pass preprocessing checks, split-definition review, baseline evaluation, and PatchCore evaluation before a model is selected for release.

### Initial MVP Data Strategy

- Dataset family: MVTec AD.
- Category: `bottle`.
- Model task: image-level anomaly detection and spatial anomaly localization.
- Fitting source: normal images from `bottle/train/good`.
- Fitting allocation: 80 percent of the normal training images.
- Validation allocation: 20 percent of the normal training images.
- Split generation: deterministic with a recorded seed.
- Validation content: normal images only.
- Initial threshold strategy: derived only from normal validation scores.
- Final evaluation source: the complete official `bottle/test` directory.
- Defect-type directory names are used for grouped evaluation only and are not learned output classes.
- Ground-truth masks are used for final localization evaluation and are not used to construct the normal Memory Bank.

#### Generated Split Manifest

The initial split has been generated and verified:

- Manifest: `configs/splits/mvtec-ad-bottle-seed-42.json`.
- Schema version: 1.
- Split seed: 42.
- Source images: 209.
- Fitting images: 167.
- Normal validation images: 42.
- Overlapping entries: 0.
- Source coverage: all 209 images are assigned exactly once.
- Stored paths: repository-independent paths relative to the MVTec AD dataset root.

The manifest is the authoritative split definition for the first MVP experiment. Model-development code must consume this manifest rather than generating a new split implicitly.

This strategy preserves the official test set for final evaluation. This strategy preserves the official test set for final evaluation. The split manifest, seed, and resulting image counts are now recorded. The exact threshold method must still be finalized and recorded before final evaluation begins.

## Selection Evidence Template

The final selection record should contain:

```text
Dataset family:
Dataset category:
Dataset acquisition date:
Local archive checksum:
License reviewed:
Inventory report:
Validation report:
Normal training image count:
Validation approach:
Final test isolation confirmed:
Image dimensions:
Estimated full Memory Bank size:
CPU preprocessing estimate:
Mask availability:
Qualitative strengths:
Known limitations:
Selection rationale:
Decision date:
Related source commit:
```

## Publication Policy

Before a README, user guide, release, model card, or portfolio page includes dataset content, the project must determine whether that specific content may be redistributed.

Until this review is complete:

- do not commit original dataset images or masks;
- do not publish screenshots containing dataset images;
- do not publish archives or extracted subsets;
- do not assume that a trained Memory Bank is unrestricted merely because it is not directly viewable as an image;
- prefer diagrams, artificial tensors, self-created images, or clearly permitted materials for public documentation.

Repository documentation may link to official dataset pages and describe aggregate facts with appropriate attribution.

## Citation Records

The final project documentation and Model Card must preserve the citations requested on the official dataset page for every dataset actually used.

The corresponding publication records should be copied from the official pages at the time a dataset is selected, then checked for author names, title, venue, year, and DOI before publication.

No publication citation is treated as finalized in this document while dataset selection remains open.

## Current Decision Status

| Decision | Status |
| --- | --- |
| Candidate dataset families | MVTec AD, MVTec LOCO AD, and MVTec AD 2 |
| Downloads | MVTec AD, MVTec LOCO AD, and MVTec AD 2 complete |
| Archive checksums | All three candidate archives recorded locally |
| Extraction validation | All three candidate dataset families passed local validation |
| Dataset inventory | All three candidate dataset inventories completed |
| First dataset family | MVTec AD selected for the first MVP cycle |
| First category | `bottle` selected |
| Validation strategy | Generated deterministic 167/42 split of `bottle/train/good` using seed 42; stored in `configs/splits/mvtec-ad-bottle-seed-42.json` |
| Screenshot redistribution review | Open |
| Dataset-derived artifact redistribution review | Open |

## Immediate Next Steps

1. Complete all dataset downloads.
2. Record archive filenames, sizes, acquisition dates, and locally calculated checksums.
3. Extract each archive into a separate read-only source location.
4. Inspect the top-level structures without modifying them.
5. Implement a dataset-inventory command.
6. Generate machine-readable inventory reports.
7. Review representative normal images, anomalies, and masks locally.
8. Compare candidate categories against the MVP selection criteria.
9. Define and document the validation strategy.
10. Record the selected family and category through an explicit decision.

## Related Documentation

- `ProjectSpecification.md` defines dataset-related requirements and acceptance criteria.
- `ModelDevelopmentStrategy.md` defines split isolation, evaluation, and model-selection methodology.
- `ArchitectureOverview.md` defines the planned data and artifact boundaries.
- `DevelopmentStatus.md` records which dataset activities have actually been completed.
- `ModelCard.md` will identify the dataset and evaluation evidence used for a released model.
