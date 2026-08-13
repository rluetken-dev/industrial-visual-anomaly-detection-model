# Industrial Visual Anomaly Detection

Industrial Visual Anomaly Detection is an evolving computer-vision portfolio project for detecting and spatially localizing unusual visual patterns in industrial inspection images.

The project is designed to combine Python-based model development with portable ONNX inference artifacts and a planned .NET application stack. The first model-development cycle uses the MVTec AD `bottle` category and targets CPU-compatible anomaly detection with explainable heatmap output.

> **Development status:** The technical feature-extraction and ONNX feasibility spike is complete. Three candidate dataset families have been validated locally, the first MVP dataset and category have been selected, and a deterministic fitting/validation split has been generated. No anomaly-detection model has yet been fitted or evaluated. The backend, web client, and desktop client are planned but not implemented.

## Project Goals

The project is intended to demonstrate:

- industrial visual anomaly detection using normal-only fitting data;
- reuse of pretrained computer-vision features;
- image-level anomaly scoring;
- spatial anomaly localization through heatmaps;
- reproducible dataset validation, splitting, experimentation, and evaluation;
- portable inference artifacts using ONNX and framework-neutral supporting data;
- cross-runtime numerical validation between Python, ONNX Runtime, and .NET;
- a client-neutral ASP.NET Core inference backend;
- separate React web and WPF desktop clients;
- transparent documentation of model limitations and evaluation evidence.

## Current Verified State

The project has verified:

- Python 3.12 development inside an isolated virtual environment;
- CPU-based PyTorch and TorchVision execution;
- loading pretrained ResNet18 weights;
- extraction of intermediate `layer2` and `layer3` feature maps;
- multi-scale feature alignment and local patch-embedding generation;
- ONNX export of the feature-extractor wrapper;
- ONNX structural validation and ONNX Runtime execution;
- close numerical agreement between PyTorch and ONNX Runtime feature outputs;
- local structural, readability, inventory, and mask validation for MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- selection of MVTec AD `bottle` for the first MVP model-development cycle;
- a deterministic 167/42 fitting and normal-validation split using seed 42;
- successful execution and inspection of the TorchVision preprocessing transform associated with the default pretrained ResNet18 weights on a real `bottle` image.

The feasibility tests do not yet demonstrate anomaly-detection accuracy, localization quality, threshold quality, robustness, or production readiness.

## Initial MVP Scope

The first model-development cycle uses:

| Item | Initial decision |
| --- | --- |
| Dataset family | MVTec AD |
| Category | `bottle` |
| Task | Image-level anomaly detection and spatial localization |
| Fitting data | 167 normal images from `bottle/train/good` |
| Validation data | 42 held-out normal images from `bottle/train/good` |
| Final test data | Complete official `bottle/test` directory |
| Split seed | 42 |
| Feature extractor | Pretrained ResNet18 feasibility baseline |
| Primary strategy | PatchCore-style anomaly detection |
| Initial runtime target | CPU on Windows |
| Intended decision | `Normal` or `Anomalous` |

Defect directory names such as `broken_large`, `broken_small`, and `contamination` are used for grouped evaluation only. The initial anomaly model will not classify the defect type.

## Planned System

The intended end-to-end system consists of:

```text
Industrial inspection image
        ↓
Deterministic preprocessing
        ↓
ONNX feature extractor
        ↓
Patch embeddings
        ↓
Memory Bank distance scoring
        ↓
Image anomaly score and heatmap
        ↓
ASP.NET Core inference API
        ↓
React web client and WPF desktop client
```

Python remains responsible for model development, experiment evaluation, and artifact export. The planned .NET backend will consume validated, versioned artifacts and expose a client-neutral inference contract. Web and desktop clients will call the backend instead of duplicating model logic.

Repository boundaries for the future backend and clients remain open. This repository currently contains the model-development work.

## Model Strategy

PatchCore is the preferred primary model for the first cycle because it can:

- fit from normal images only;
- reuse pretrained visual representations;
- avoid full backbone retraining;
- produce local anomaly scores;
- support both image-level decisions and heatmaps;
- remain practical for a CPU-oriented proof of concept.

In simplified form:

```text
Normal fitting images
        ↓
Pretrained feature extractor
        ↓
Normal local feature vectors
        ↓
Memory Bank representing normal appearance
```

During inference, local features from a new image are compared with the stored normal features. Large distances indicate visually unusual regions.

A simpler baseline will be implemented before the PatchCore-style model is selected. Model and threshold decisions will use validation evidence rather than final test labels.

## Dataset Policy

The considered datasets are:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

The official MVTec pages state that these datasets are licensed under CC BY-NC-SA 4.0 and prohibit commercial use without appropriate permission.

Original archives, extracted images, masks, large derived data, and local validation reports are not stored in this repository. Dataset roots are supplied through command-line arguments or later configuration. The repository contains only source code, documentation, and small reproducibility manifests.

Do not publish original dataset images, masks, adapted samples, Memory Banks, or other dataset-derived artifacts without reviewing the applicable license and redistribution conditions.

## Repository Structure

```text
industrial-visual-anomaly-detection-model/
├── configs/
│   └── splits/
│       └── mvtec-ad-bottle-seed-42.json
├── docs/
│   ├── ArchitectureOverview.md
│   ├── DatasetDocumentation.md
│   ├── DevelopmentStatus.md
│   ├── ModelDevelopmentStrategy.md
│   └── ProjectSpecification.md
├── scripts/
│   ├── create_mvtec_ad_split.py
│   ├── inspect_preprocessing.py
│   ├── validate_mvtec_ad.py
│   ├── validate_mvtec_ad_2.py
│   └── validate_mvtec_loco_ad.py
├── .gitignore
├── .python-version
├── environment_check.py
├── README.md
└── requirements.txt
```

Generated ONNX files and local datasets are intentionally excluded from Git.

## Technology Stack

Current model-development stack:

- Python 3.12.10
- PyTorch 2.13.0 CPU build
- TorchVision 0.28.0 CPU build
- Pillow 12.2.0
- NumPy 2.4.4
- ONNX 1.22.0
- ONNX Runtime 1.28.0
- ONNXScript 0.7.1

Planned application stack:

- ASP.NET Core backend
- ONNX Runtime for .NET
- React and TypeScript web client
- WPF desktop client
- automated test and CI workflows

The planned stack must not be interpreted as implemented functionality.

## Local Development Setup

### Prerequisites

- Windows development environment
- Python 3.12.10
- Git
- sufficient local storage for separately downloaded datasets

### Create The Virtual Environment

From the repository root:

```powershell
python -m venv .venv
```

PowerShell script execution policies may prevent activation. Activation is optional; every command can call the environment interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

If local policy permits activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Verify The Environment

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe .\environment_check.py
```

The technical spike currently exports generated ONNX files into the working directory. These files are ignored by Git.

## Dataset Validation

Datasets must be downloaded directly from their official sources and stored outside the repository.

Example validator commands:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_mvtec_ad.py `
    --dataset-root C:\path\to\mvtec-ad

.\.venv\Scripts\python.exe .\scripts\validate_mvtec_loco_ad.py `
    --dataset-root C:\path\to\mvtec-loco-ad

.\.venv\Scripts\python.exe .\scripts\validate_mvtec_ad_2.py `
    --dataset-root C:\path\to\mvtec_ad_2
```

The full validators decode every PNG file and may take several minutes for the larger datasets. They do not modify source images.

Detailed acquisition checksums, inventory results, validation findings, licensing notes, and the MVTec LOCO AD count discrepancy are documented in [`docs/DatasetDocumentation.md`](docs/DatasetDocumentation.md).

## Split Manifest

The first MVP split is stored in:

```text
configs/splits/mvtec-ad-bottle-seed-42.json
```

It assigns all 209 normal `bottle/train/good` images exactly once:

- 167 images for fitting;
- 42 images for normal validation;
- 0 overlapping entries.

The manifest is authoritative. Model code must consume it rather than generating another split implicitly.

The split can be regenerated deliberately with:

```powershell
.\.venv\Scripts\python.exe .\scripts\create_mvtec_ad_split.py `
    --dataset-root C:\path\to\mvtec-ad `
    --output .\configs\splits\mvtec-ad-bottle-seed-42.json
```

Regeneration should be followed by review of the Git diff. An unexpected manifest change must not be accepted silently.

## Preprocessing Inspection

Inspect the verified ResNet18 preprocessing contract for one image:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_preprocessing.py `
    --image C:\path\to\mvtec-ad\bottle\train\good\000.png
```

The current feasibility configuration converts images to RGB and applies the transform associated with the default pretrained ResNet18 weights:

```text
Resize to 256
→ center crop to 224 × 224
→ convert to a Float32 tensor
→ normalize with ImageNet channel statistics
```

This remains an MVP preprocessing baseline. Higher input resolutions and alternative crop behavior require validation evidence before adoption.

## Documentation

- [Project Specification](docs/ProjectSpecification.md)
- [Architecture Overview](docs/ArchitectureOverview.md)
- [Development Status](docs/DevelopmentStatus.md)
- [Model Development Strategy](docs/ModelDevelopmentStrategy.md)
- [Dataset Documentation](docs/DatasetDocumentation.md)

Future documentation will include experiment reports, a User Guide, and a Model Card for a selected evaluated artifact.

## Current Limitations

- No anomaly-detection model has been fitted.
- No baseline or PatchCore evaluation result exists.
- The current ONNX artifact is a generated feature-extraction feasibility artifact, not a released anomaly model.
- Only CPU-based local PyTorch execution has been verified.
- The initial preprocessing resolution may lose small visual defects.
- No ASP.NET Core backend has been implemented.
- No React web client has been implemented.
- No WPF desktop client has been implemented.
- Dataset screenshot and dataset-derived artifact redistribution remain under review.
- The project has not been validated for production, safety-critical, pharmaceutical, or regulatory use.

## Roadmap

- persist machine-readable dataset-validation reports;
- visually inspect the resized and cropped model input;
- refactor the technical spike into testable modules;
- implement deterministic preprocessing backed by the split manifest;
- implement and evaluate a simple baseline;
- implement the initial PatchCore-style Memory Bank;
- define thresholding from normal validation scores;
- conduct a locked final MVTec AD `bottle` evaluation;
- export and version the selected artifact package;
- validate Python, ONNX Runtime, and .NET inference parity;
- implement the inference backend;
- implement separate web and desktop clients;
- add automated tests, CI, screenshots, and release documentation.

## Responsible Use

This project is an educational and portfolio-oriented demonstration. It must not be represented as a certified quality-control system and must not make autonomous production acceptance, safety, or regulatory decisions.

## License

No repository license has been selected yet. Until a license is added, default copyright restrictions apply to the source code. Third-party datasets, pretrained weights, libraries, and generated artifacts remain subject to their own licenses and terms.
