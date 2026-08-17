# Industrial Visual Anomaly Detection

[![CI](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rluetken-dev/industrial-visual-anomaly-detection-model?include_prereleases)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/releases)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

Industrial Visual Anomaly Detection is an educational and portfolio-oriented computer-vision project for detecting unusual visual patterns in industrial inspection images and highlighting suspicious regions.

The implemented Python MVP uses a frozen pretrained ResNet18 and a PatchCore-inspired feature-memory approach. It can validate datasets, create reproducible data splits, build category-specific anomaly models, evaluate them, generate heatmaps, export reusable artifacts, classify individual images on CPU, and expose a loaded artifact through an internal FastAPI inference service.

> **Current status:** The Python model-development, artifact export, local inference, and internal HTTP inference-service MVPs are implemented. Bottle and Capsule have been evaluated exploratorily, a reusable Capsule reference artifact can be exported, and 68 automated tests cover the main deterministic and service components. The internal prediction response now includes a threshold-normalized anomaly heatmap as a Base64-encoded PNG. End-to-end communication from the separate ASP.NET Core backend through the Python service to the model has been verified locally.

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
- reusable file-path and binary-stream inference APIs;
- single-image inference CLI;
- internal FastAPI inference service with startup-time artifact loading;
- multipart image prediction endpoint for backend integration;
- threshold-normalized anomaly heatmaps returned as Base64-encoded RGB PNG images;
- automated unit and service tests;
- GitHub Actions CI.

## System Integration

The Python repository owns model development and inference execution. A separate ASP.NET Core backend owns the public application API and delegates model execution to this internal Python service.

```text
Client
  -> ASP.NET Core backend
  -> internal FastAPI inference service
  -> loaded PyTorch artifact and feature memory
  -> prediction response
  -> ASP.NET Core response
```

The service loads the configured artifact and creates the feature extractor once during application startup. Requests reuse both objects instead of loading the approximately 410 MiB Capsule feature memory for every image.

The verified local integration uses:

- backend endpoint: `POST /api/v1/analyses`;
- internal Python endpoint: `POST /api/v1/predictions`;
- multipart upload field: `image`;
- Python service address: `http://127.0.0.1:8000`.

The ASP.NET Core backend is maintained in the separate [industrial-visual-anomaly-detection-backend](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-backend) repository.

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
│       ├── service/
│       │   ├── app.py
│       │   ├── heatmap_encoding.py
│       │   ├── prediction_response.py
│       │   ├── prediction_routes.py
│       │   ├── runtime.py
│       │   └── settings.py
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
- FastAPI 0.139.2
- Uvicorn 0.50.0
- python-multipart 0.0.32
- HTTPX 0.28.1
- Pillow 12.2.0
- NumPy 2.4.4
- ONNX 1.22.0
- ONNX Runtime 1.28.0
- ONNXScript 0.7.1
- GitHub Actions

Related application stack:

- separate ASP.NET Core backend with verified Python-service integration;
- separate WPF desktop client with a verified image-analysis workflow;
- planned separate web client;
- versioned HTTP inference boundaries between the clients, backend, and Python service.

## Local Setup

### Prerequisites

- Python 3.12
- Git
- sufficient storage for datasets and generated model artifacts outside Git

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

The current suite contains 68 tests. The GitHub Actions workflow runs equivalent checks with Python 3.12 on Ubuntu for every push and pull request targeting `main`.

## Dataset Setup

Download datasets directly from their official sources and store them outside the repository:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

Example dataset root:

```text
C:\path\to\industrial-visual-anomaly-detection\raw\
```

Supply your own dataset root to every dataset-dependent command.

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

The current format is a versioned Python/PyTorch artifact, not a framework-neutral production package.

## Predict One Image from the CLI

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_image.py `
    --artifact .\outputs\model-artifacts\mvtec-ad-capsule-320 `
    --image C:\path\to\image.png
```

The command prints the model configuration, anomaly score, threshold, decision, feature-memory shape, and relevant timings.

## Run the Internal Inference Service

Configure the artifact and optional nearest-neighbor chunk size:

```powershell
$env:IVAD_MODEL_ARTIFACT = "$PWD\outputs\model-artifacts\mvtec-ad-capsule-320"
$env:IVAD_MEMORY_CHUNK_SIZE = "4096"
```

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
    industrial_visual_anomaly_detection.service.app:app `
    --host 127.0.0.1 `
    --port 8000
```

The startup process loads the configured artifact and creates the frozen feature extractor once. Keep the process running while the backend sends inference requests.

### Check Service Health

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/health/live `
    -Method Get
```

Expected response:

```json
{
  "status": "healthy"
}
```

### Request a Prediction

```powershell
curl.exe `
    -X POST `
    http://127.0.0.1:8000/api/v1/predictions `
    -F "image=@C:\path\to\image.png;type=image/png"
```

Example response:

```json
{
  "modelId": "mvtec-ad-capsule-320",
  "category": "capsule",
  "score": 4.992109298706055,
  "threshold": 2.501821517944336,
  "isAnomalous": true,
  "heatmap": {
    "contentType": "image/png",
    "width": 320,
    "height": 320,
    "dataBase64": "<Base64-encoded PNG data>"
  }
}
```

The heatmap is threshold-normalized, resized to the configured model input dimensions, colorized as an RGB image, encoded as PNG, and transported as Base64 text. The ASP.NET Core backend may forward or transform this internal representation for application clients.

## Visualization

The project can convert the patch-score grid into a resized heatmap and overlay it on the input image. It supports:

- per-image normalization for qualitative inspection;
- threshold-based normalization for better cross-image comparison;
- configurable heatmap opacity;
- threshold-normalized RGB heatmap generation for service responses;
- Base64-encoded PNG transport through the internal prediction endpoint.

The service-generated heatmap uses the model threshold as its fixed normalization reference and is resized to the configured model input dimensions. This makes its color scale more comparable across images than independent per-image normalization.

Heatmaps are explanation aids. Pixel-level benchmark metrics against ground-truth masks are not yet implemented.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`. It:

- installs Python 3.12;
- installs pinned dependencies;
- installs the project in editable mode;
- compiles Python source files;
- checks installed dependencies;
- runs all unit and service tests.

Dataset-dependent benchmarks, real service startup with a large artifact, and artifact exports are intentionally excluded from CI because the licensed datasets and generated feature memories are not stored in the repository.

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
- the current artifact uses PyTorch tensor serialization and is trusted local input;
- artifact metadata does not yet fully describe every preprocessing operation;
- one category-specific artifact must not be assumed to generalize to another category;
- the inference service currently serializes access to its shared model runtime;
- service authentication, containerization, production health checks, and deployment hardening are not implemented;
- the separate ASP.NET Core backend and WPF desktop client provide verified MVP integration but are not production deployments;
- the web client is not implemented;
- Base64-encoded heatmaps increase the size of internal prediction responses;
- the system is not certified for production quality-control decisions.

## Roadmap

- propagate service-generated heatmaps through the ASP.NET Core backend to the WPF desktop client;
- add multi-artifact loading or model selection for categories such as Bottle and Capsule;
- add backend-to-service integration coverage suitable for CI;
- evaluate at least one MVTec AD category beyond Bottle and Capsule;
- implement pixel-level localization metrics;
- define an evaluation protocol that does not tune on inspected test data;
- investigate principled feature-memory reduction and faster nearest-neighbor search;
- complete portable artifact and 320 × 320 ONNX parity work;
- implement the separate web client;
- add a Model Card and updated release documentation.

## Dataset and Artifact Policy

The MVTec datasets are published under CC BY-NC-SA 4.0 and restrict commercial use. Original images, masks, screenshots, extracted subsets, feature memories, and model artifacts are not published by this repository unless their redistribution has been reviewed separately.

## Responsible Use

This project is an experimental educational demonstration. It must not be represented as a certified inspection system or used autonomously for production acceptance, safety, medical, or regulatory decisions.

## Repository License

No source-code license has been selected yet. Until a license is added, default copyright restrictions apply to the repository source. Datasets, pretrained weights, dependencies, and generated artifacts remain subject to their own licenses and terms.
