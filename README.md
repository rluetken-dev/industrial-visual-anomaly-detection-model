# Industrial Visual Anomaly Detection

[![CI](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rluetken-dev/industrial-visual-anomaly-detection-model?include_prereleases)](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-model/releases)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

Industrial Visual Anomaly Detection is an educational and portfolio-oriented computer-vision project for detecting unusual visual patterns in industrial inspection images and highlighting suspicious regions.

The Python implementation uses a frozen pretrained ResNet18 and a PatchCore-inspired feature-memory approach. It can train category-specific anomaly models from normal images, evaluate them, export reusable artifacts, classify images on CPU, generate heatmaps, and expose multiple loaded artifacts through an internal FastAPI inference service.

> **Current status:** Model development, generalized artifact export, dataset-independent evaluation, local inference, and registry-based multi-model HTTP inference are implemented. A model registry can define multiple enabled artifacts and one default model. Callers can discover the catalog and select a model per prediction request. The current suite contains 144 automated test methods covering deterministic data preparation, training, artifacts, evaluation, inference, registry validation, runtime selection, and service endpoints.

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

The output is `normal` or `anomalous`. It does not classify the exact defect type.

## Implemented Capabilities

- validation tooling for MVTec AD, MVTec LOCO AD, and MVTec AD 2;
- recursive discovery of normal PNG and JPEG images from external directories;
- deterministic fitting and validation splits;
- portable `training_split.json` records with relative image paths;
- dataset-independent model training and artifact export;
- configurable deterministic image preprocessing;
- frozen ResNet18 `layer2` and `layer3` feature extraction;
- 384-dimensional multi-scale patch embeddings;
- complete or deterministically sampled feature memories;
- exact chunked Euclidean nearest-neighbor scoring;
- maximum and top-fraction-mean image-score aggregation;
- configurable normal-validation quantile thresholds;
- dataset-independent labeled-image evaluation through CSV manifests;
- image-level metrics, score distributions, confusion matrices, and group reporting;
- anomaly heatmaps and image overlays;
- schema-versioned Python/PyTorch artifact export and loading;
- compatibility loading for schema-version-1 artifacts;
- file-path and binary-stream inference APIs;
- single-image inference CLI;
- legacy single-artifact service startup;
- registry-based startup with multiple enabled artifacts;
- startup validation of model identifiers, defaults, paths, and registry structure;
- model-catalog endpoint;
- optional per-request model selection;
- multipart image prediction endpoint;
- threshold-normalized heatmaps returned as Base64-encoded RGB PNG images;
- automated unit and service tests;
- GitHub Actions CI.

## System Integration

The Python repository owns model development, the model registry contract, and inference execution. A separate ASP.NET Core backend owns the public application API and delegates execution to this internal service.

```text
Desktop or other client
  -> ASP.NET Core backend
  -> internal FastAPI inference service
  -> model registry
  -> selected loaded artifact and feature memory
  -> prediction response
  -> ASP.NET Core response
```

In registry mode, every enabled artifact and its runtime are loaded once during startup. Requests reuse those objects rather than rebuilding the feature extractor or repeatedly loading large feature memories.

The integration boundaries are:

- backend endpoint: `POST /api/v1/analyses`;
- internal catalog endpoint: `GET /api/v1/models`;
- internal prediction endpoint: `POST /api/v1/predictions`;
- multipart image field: `image`;
- optional multipart model field: `modelId`;
- local Python service address: `http://127.0.0.1:8000`.

Related repositories:

- [ASP.NET Core backend](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-backend)
- [Docker Compose stack](https://github.com/rluetken-dev/industrial-visual-anomaly-detection-stack)

## Reference Models

The verified local registry has been exercised with:

```text
mvtec-ad-capsule-320
mvtec-ad-bottle-generalized-320
visa-candle-generalized-q95-320
visa-cashew-generalized-q95-320
```

These generated artifacts remain outside Git. Each artifact is category-specific; a model trained for one product category must not be assumed to generalize to another.

### Capsule Reference

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

These results are exploratory rather than an untouched final benchmark because test images influenced development analysis.

### VisA Candle Calibration

Normal-validation quantiles of 1.00, 0.99, and 0.95 were compared with identical feature memories. The q95 candidate achieved the best observed F1 trade-off in that experiment, but it requires independent confirmation because official test images were inspected during selection.

See [VisA Candle Threshold Calibration](docs/experiments/visa-candle-threshold-calibration.md) for the complete experiment record.

## Repository Structure

```text
industrial-visual-anomaly-detection-model/
├── .github/workflows/ci.yml
├── configs/splits/
├── docs/
├── scripts/
│   ├── evaluate_model_artifact.py
│   ├── evaluate_mvtec_ad_category.py
│   ├── export_image_directory_model.py
│   ├── export_mvtec_ad_model.py
│   ├── predict_image.py
│   └── validate_*.py
├── src/industrial_visual_anomaly_detection/
│   ├── artifacts/
│   ├── datasets/
│   ├── models/
│   ├── service/
│   │   ├── app.py
│   │   ├── model_registry_config.py
│   │   ├── model_routes.py
│   │   ├── prediction_routes.py
│   │   ├── runtime.py
│   │   ├── runtime_registry.py
│   │   └── settings.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── training.py
│   └── visualization.py
├── tests/
├── COMMITS.md
├── environment_check.py
├── pyproject.toml
└── requirements.txt
```

Local datasets, generated reports, heatmaps, feature memories, registries, and model artifacts are excluded from Git.

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

## Local Setup

### Prerequisites

- Python 3.12
- Git
- sufficient storage for datasets and generated artifacts outside Git

The current implementation supports CPU-only execution. A CUDA-capable GPU is not required.

### Create and Install the Environment

```powershell
python -m venv .venv

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

The current suite contains 144 test methods. GitHub Actions runs equivalent checks with Python 3.12 on Ubuntu for pushes and pull requests targeting `main`.

## Dataset Setup

Download datasets from their official sources and store them outside the repository:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- [VisA](https://registry.opendata.aws/visa/)

Supply your own dataset root to dataset-dependent commands.

## Export a Model Artifact

### MVTec AD Manifest Workflow

```powershell
.\.venv\Scripts\python.exe .\scripts\export_mvtec_ad_model.py `
    --dataset-root C:\path\to\mvtec-ad `
    --manifest .\configs\splits\mvtec-ad-capsule-seed-42.json `
    --output-directory .\outputs\model-artifacts\mvtec-ad-capsule-320 `
    --input-size 320 `
    --top-fraction 0.01 `
    --threshold-quantile 0.95 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

### Normal-Image Directory Workflow

```powershell
.\.venv\Scripts\python.exe .\scripts\export_image_directory_model.py `
    --image-directory C:\path\to\normal-images `
    --dataset custom-dataset `
    --category custom-category `
    --output-directory .\outputs\model-artifacts\custom-category-320 `
    --validation-fraction 0.2 `
    --split-seed 42 `
    --input-size 320 `
    --top-fraction 0.01 `
    --threshold-quantile 0.95 `
    --memory-fraction 1.0 `
    --sampling-seed 42
```

The exporter writes:

```text
metadata.json
feature_memory.pt
training_split.json
```

New generalized exports use schema version 2 and record `threshold_method` and `threshold_quantile`. The loader remains compatible with schema-version-1 artifacts by applying the former maximum-normal defaults.

## Evaluate an Exported Artifact

The dataset-independent evaluator accepts a CSV manifest:

```csv
image,group,is_anomalous
relative/path/to/normal-image.jpg,normal,false
relative/path/to/anomalous-image.jpg,anomaly,true
```

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_model_artifact.py `
    --artifact .\outputs\model-artifacts\visa-candle-generalized-q95-320 `
    --dataset-root C:\path\to\visa\extracted `
    --manifest C:\path\to\visa\evaluation_manifest.csv
```

## Predict One Image from the CLI

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_image.py `
    --artifact .\outputs\model-artifacts\mvtec-ad-capsule-320 `
    --image C:\path\to\image.png
```

## Run the Internal Inference Service

The service supports two mutually exclusive startup modes.

### Registry Mode

Create `models.json` in the artifact root:

```json
{
  "schemaVersion": 1,
  "defaultModelId": "mvtec-ad-capsule-320",
  "models": [
    {
      "id": "mvtec-ad-capsule-320",
      "displayName": "MVTec AD - Capsule",
      "artifactDirectory": "mvtec-ad-capsule-320",
      "enabled": true
    },
    {
      "id": "visa-cashew-generalized-q95-320",
      "displayName": "VisA - Cashew",
      "artifactDirectory": "visa-cashew-generalized-q95-320",
      "enabled": true
    }
  ]
}
```

Artifact directories are resolved relative to the registry file. Configure the service:

```powershell
$env:IVAD_MODEL_REGISTRY = "$PWD\outputs\model-artifacts\models.json"
Remove-Item Env:IVAD_MODEL_ARTIFACT -ErrorAction SilentlyContinue
$env:IVAD_MEMORY_CHUNK_SIZE = "4096"
```

### Legacy Single-Artifact Mode

```powershell
$env:IVAD_MODEL_ARTIFACT = "$PWD\outputs\model-artifacts\mvtec-ad-capsule-320"
Remove-Item Env:IVAD_MODEL_REGISTRY -ErrorAction SilentlyContinue
$env:IVAD_MEMORY_CHUNK_SIZE = "4096"
```

Exactly one of `IVAD_MODEL_REGISTRY` and `IVAD_MODEL_ARTIFACT` must be configured.

### Start the Service

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
    industrial_visual_anomaly_detection.service.app:app `
    --host 127.0.0.1 `
    --port 8000
```

### Check Health

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/health/live `
    -Method Get
```

### Retrieve the Model Catalog

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/api/v1/models `
    -Method Get |
    ConvertTo-Json -Depth 5
```

### Request a Model-Specific Prediction

```powershell
curl.exe `
    --request POST `
    http://127.0.0.1:8000/api/v1/predictions `
    --form "image=@C:\path\to\image.png;type=image/png" `
    --form "modelId=mvtec-ad-capsule-320"
```

If `modelId` is omitted in registry mode, the configured default model is used. An unknown model identifier returns a clear not-found response.

Example response:

```json
{
  "modelId": "mvtec-ad-capsule-320",
  "category": "capsule",
  "score": 4.992109,
  "threshold": 2.501822,
  "isAnomalous": true,
  "heatmap": {
    "contentType": "image/png",
    "width": 320,
    "height": 320,
    "dataBase64": "<Base64-encoded PNG data>"
  }
}
```

## CI

`.github/workflows/ci.yml` installs Python 3.12 and pinned dependencies, installs the project, compiles Python sources, checks installed dependencies, and runs all automated tests.

Dataset-dependent benchmarks, real startup with large artifacts, and artifact exports are excluded because licensed datasets and generated feature memories are not stored in the repository.

## Documentation

- [Architecture Overview](docs/ArchitectureOverview.md)
- [VisA Candle Threshold Calibration](docs/experiments/visa-candle-threshold-calibration.md)
- [Dataset Documentation](docs/DatasetDocumentation.md)
- [Development Status](docs/DevelopmentStatus.md)
- [Model Development Strategy](docs/ModelDevelopmentStrategy.md)
- [Project Specification](docs/ProjectSpecification.md)
- [Commit Message Guidelines](COMMITS.md)

## Known Limitations

- benchmark results are exploratory because test images influenced development;
- the provisional VisA Candle q95 threshold requires independent confirmation;
- pixel-level localization metrics are not implemented;
- exact nearest-neighbor search is computationally and memory intensive;
- all enabled registry models and feature memories are loaded during startup;
- dynamic registry reload and lazy model loading are not implemented;
- the current artifact uses trusted local PyTorch tensor serialization;
- artifact metadata does not yet fully describe every preprocessing operation;
- the inference service serializes access to each shared runtime;
- Base64-encoded heatmaps increase response size;
- service authentication and production deployment hardening are not implemented;
- the system is not certified for production quality-control decisions.

## Roadmap

- publish a registry-capable model-service release;
- validate thresholds on previously unused data;
- add backend-to-service integration coverage suitable for CI;
- implement pixel-level localization metrics;
- define a strict calibration and final-test protocol;
- investigate principled feature-memory reduction and faster nearest-neighbor search;
- evaluate lazy model loading and controlled registry reload;
- complete portable artifact and ONNX parity work;
- add a Model Card and updated release documentation.

## Dataset and Artifact Policy

The MVTec datasets are published under CC BY-NC-SA 4.0 and restrict commercial use. Original images, masks, screenshots, extracted subsets, feature memories, registries, and model artifacts are not published by this repository unless their redistribution has been reviewed separately.

## Responsible Use

This project is an experimental educational demonstration. It must not be represented as a certified inspection system or used autonomously for production acceptance, safety, medical, or regulatory decisions.

## Repository License

No source-code license has been selected yet. Until a license is added, default copyright restrictions apply to the repository source. Datasets, pretrained weights, dependencies, and generated artifacts remain subject to their own licenses and terms.
