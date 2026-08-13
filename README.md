# Industrial Visual Anomaly Detection

[![CI](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rluetken-dev/industrial-visual-anomaly-detection-model?include_prereleases)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/releases)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

Industrial Visual Anomaly Detection is an educational and portfolio-oriented computer-vision project for detecting unusual visual patterns in industrial inspection images and highlighting suspicious regions.

The implemented Python MVP uses a frozen pretrained ResNet18 and a PatchCore-inspired feature-memory approach. It can validate datasets, create reproducible data splits, build category-specific anomaly models, evaluate them, generate heatmaps, export reusable artifacts, and classify individual images on CPU.

> **Current status:** The Python model-development and local inference MVP is implemented. Bottle and Capsule have been evaluated exploratorily, a reusable Capsule reference artifact can be exported, and 54 automated tests cover the main deterministic components. The ASP.NET Core backend, web client, and desktop client are planned but not yet implemented.

## What the Model Does

The model learns only from normal images:

1. images are converted to RGB, resized, and normalized;
2. a frozen pretrained ResNet18 extracts intermediate feature maps;
3. `layer2` and `layer3` features are combined into local patch embeddings;
4. embeddings from normal fitting images form a feature memory;
5. every patch of a new image is compared with its nearest normal neighbor;
6. large distances indicate unusual visual regions;
7. patch scores form an anomaly map and an image-level score;
8. the score is compared with a threshold derived from held-out normal images.

The current output is `normal` or `anomalous`. It does not classify the exact defect type.

## Implemented Capabilities

- validation tooling for MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- optional schema-versioned JSON validation reports;
- deterministic fitting and validation manifests;
- configurable deterministic image preprocessing;
- frozen ResNet18 `layer2` and `layer3` feature extraction;
- 384-dimensional multi-scale patch embeddings;
- complete feature-memory construction;
- optional deterministic random feature-memory sampling;
- exact chunked Euclidean nearest-neighbor scoring;
- maximum and top-fraction-mean image-score aggregation;
- normal-validation-based threshold selection;
- image-level evaluation metrics and defect-group reporting;
- anomaly heatmaps and image overlays;
- versioned Python/PyTorch model artifact export and loading;
- reusable single-image inference API and CLI;
- automated unit tests and GitHub Actions CI.

## Reference Configurations

### Bottle Baseline

| Item | Value |
| --- | --- |
| Dataset | MVTec AD |
| Category | `bottle` |
| Input size | 224 × 224 |
| Fitting images | 167 normal images |
| Validation images | 42 normal images |
| Split seed | 42 |
| Patch grid | 28 × 28 |
| Embedding dimension | 384 |
| Feature memory | Complete fitting memory |

The top-1%-patch mean achieved perfect separation on the inspected Bottle test partition. This is an exploratory result, not an untouched final benchmark, because test images influenced development analysis.

### Capsule Artifact Reference

| Item | Value |
| --- | --- |
| Dataset | MVTec AD |
| Category | `capsule` |
| Input size | 320 × 320 |
| Fitting images | 175 normal images |
| Validation images | 44 normal images |
| Split seed | 42 |
| Patch grid | 40 × 40 |
| Embedding dimension | 384 |
| Aggregation | Mean of highest-scoring 1% of patches |
| Feature memory | 280,000 × 384, approximately 410.16 MiB |
| Threshold | Approximately `2.501822` |

Exploratory Capsule test results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9470 |
| Precision | 0.9811 |
| Recall | 0.9541 |
| F1 score | 0.9674 |
| False positives | 2 |
| False negatives | 5 |

## Repository Structure

```text
industrial-visual-anomaly-detection-model/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── splits/
│       ├── mvtec-ad-bottle-seed-42.json
│       └── mvtec-ad-capsule-seed-42.json
├── docs/
│   ├── ArchitectureOverview.md
│   ├── DatasetDocumentation.md
│   ├── DevelopmentStatus.md
│   ├── ModelDevelopmentStrategy.md
│   └── ProjectSpecification.md
├── scripts/
│   ├── create_mvtec_ad_split.py
│   ├── evaluate_mvtec_ad_category.py
│   ├── export_mvtec_ad_model.py
│   ├── inspect_preprocessing.py
│   ├── predict_image.py
│   ├── validate_mvtec_ad.py
│   ├── validate_mvtec_ad_2.py
│   ├── validate_mvtec_loco_ad.py
│   └── visualize_bottle_anomaly.py
├── src/
│   └── industrial_visual_anomaly_detection/
│       ├── artifacts/
│       ├── datasets/
│       ├── models/
│       ├── evaluation.py
│       ├── inference.py
│       ├── preprocessing.py
│       └── visualization.py
├── tests/
├── COMMITS.md
├── environment_check.py
├── pyproject.toml
└── requirements.txt
```

Local datasets, generated reports, heatmaps, ONNX files, feature memories, and model artifacts are excluded from Git.

## Technology Stack

- Python 3.12.10
- PyTorch 2.13.0 CPU
- TorchVision 0.28.0 CPU
- Pillow 12.2.0
- NumPy 2.4.4
- ONNX 1.22.0
- ONNX Runtime 1.28.0
- ONNXScript 0.7.1
- GitHub Actions

Planned application stack:

- ASP.NET Core backend;
- separate web client;
- separate desktop client;
- shared versioned inference API.

## Local Setup

### Prerequisites

- Python 3.12
- Git
- sufficient storage for datasets kept outside the repository

The current implementation supports CPU-only execution. A CUDA-capable GPU is not required.

### Create the Virtual Environment

```powershell
python -m venv .venv
```

Activate it when PowerShell permits script execution:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activation is optional. All commands below call the environment interpreter explicitly.

### Install Dependencies and Package

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install --editable .
```

## Run the Quality Checks

```powershell
.\.venv\Scripts\python.exe -m compileall `
    .\src `
    .\scripts `
    .\tests `
    .\environment_check.py

.\.venv\Scripts\python.exe -m pip check

.\.venv\Scripts\python.exe -m unittest discover `
    -s .\tests `
    -p "test_*.py" `
    -v
```

The GitHub Actions workflow runs equivalent checks with Python 3.12 on Ubuntu for every push and pull request targeting `main`.

## Dataset Setup

Download datasets directly from their official sources and store them outside the repository:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

The local project used:

```text
C:\dev\data\industrial-visual-anomaly-detection\raw\
```

This path is only an example. Supply your own dataset root to every command.

## Validate the Datasets

### MVTec AD

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_mvtec_ad.py `
    --dataset-root C:\path\to\mvtec-ad `
    --report .\validation-reports\mvtec-ad.json
```

### MVTec LOCO AD

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_mvtec_loco_ad.py `
    --dataset-root C:\path\to\mvtec-loco-ad `
    --report .\validation-reports\mvtec-loco-ad.json
```

### MVTec AD 2

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_mvtec_ad_2.py `
    --dataset-root C:\path\to\mvtec_ad_2 `
    --report .\validation-reports\mvtec-ad-2.json
```

Validation reports contain resolved local paths and are intentionally ignored by Git.

## Evaluate an MVTec AD Category

Bottle at 224 × 224:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_mvtec_ad_category.py `
    --dataset-root C:\path\to\mvtec-ad `
    --manifest .\configs\splits\mvtec-ad-bottle-seed-42.json `
    --input-size 224 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

Capsule at 320 × 320:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_mvtec_ad_category.py `
    --dataset-root C:\path\to\mvtec-ad `
    --manifest .\configs\splits\mvtec-ad-capsule-seed-42.json `
    --input-size 320 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

Exact nearest-neighbor evaluation on CPU can take several minutes, especially with the complete 320 × 320 feature memory.

## Export a Model Artifact

```powershell
.\.venv\Scripts\python.exe .\scripts\export_mvtec_ad_model.py `
    --dataset-root C:\path\to\mvtec-ad `
    --manifest .\configs\splits\mvtec-ad-capsule-seed-42.json `
    --output-directory .\outputs\model-artifacts\mvtec-ad-capsule-320 `
    --input-size 320 `
    --top-fraction 0.01 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

The artifact contains:

```text
metadata.json
feature_memory.pt
```

The current format is a versioned Python/PyTorch artifact, not yet a framework-neutral production package.

## Predict One Image

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_image.py `
    --artifact .\outputs\model-artifacts\mvtec-ad-capsule-320 `
    --image C:\path\to\image.png
```

The command prints the model configuration, anomaly score, threshold, decision, feature-memory shape, and relevant timings.

## Visualization

The project can convert the patch-score grid into a resized heatmap and overlay it on the input image. It supports:

- per-image normalization for qualitative inspection;
- threshold-based normalization for better cross-image comparison;
- configurable heatmap opacity.

Heatmaps are explanation aids. Pixel-level benchmark metrics against ground-truth masks are not yet implemented.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`. It:

- installs Python 3.12;
- installs pinned dependencies;
- installs the project in editable mode;
- compiles Python source files;
- checks installed dependencies;
- runs all unit tests.

Dataset-dependent benchmarks and large artifact exports are intentionally excluded from CI because the licensed datasets and generated feature memories are not stored in the repository.

## Documentation

- [Architecture Overview](docs/ArchitectureOverview.md)
- [Dataset Documentation](docs/DatasetDocumentation.md)
- [Development Status](docs/DevelopmentStatus.md)
- [Model Development Strategy](docs/ModelDevelopmentStrategy.md)
- [Project Specification](docs/ProjectSpecification.md)
- [Commit Message Guidelines](COMMITS.md)

## Known Limitations

- current benchmark results are exploratory because test images were inspected during development;
- pixel-level localization metrics are not implemented;
- exact nearest-neighbor search is computationally and memory intensive;
- the current artifact uses PyTorch tensor serialization;
- artifact metadata does not yet fully describe every preprocessing operation;
- the selected 320 × 320 pipeline still requires updated ONNX export and parity verification;
- one category-specific artifact must not be assumed to generalize to another category;
- no backend, web client, or desktop client exists yet;
- the system is not certified for production quality-control decisions.

## Roadmap

- evaluate at least one MVTec AD category beyond Bottle and Capsule;
- implement pixel-level localization metrics;
- define an evaluation protocol that does not tune on inspected test data;
- investigate principled feature-memory reduction and faster nearest-neighbor search;
- complete portable artifact and 320 × 320 ONNX parity work;
- implement the ASP.NET Core inference backend;
- implement separate web and desktop clients;
- add a Model Card and release documentation.

## Dataset and Artifact Policy

The MVTec datasets are published under CC BY-NC-SA 4.0 and restrict commercial use. Original images, masks, screenshots, extracted subsets, feature memories, and model artifacts are not published by this repository unless their redistribution has been reviewed separately.

## Responsible Use

This project is an experimental educational demonstration. It must not be represented as a certified inspection system or used autonomously for production acceptance, safety, medical, or regulatory decisions.

## Repository License

No source-code license has been selected yet. Until a license is added, default copyright restrictions apply to the repository source. Datasets, pretrained weights, dependencies, and generated artifacts remain subject to their own licenses and terms.
