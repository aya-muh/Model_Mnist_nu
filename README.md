# Optical Digit Verification System (ODVS) — MNIST Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit--Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning, MLOps, and Computer Vision system for handwritten digit recognition based on the **MNIST** benchmark dataset (70,000 images, 28&times;28 grayscale, 784 features, classes 0–9). 

The project encompasses exploratory data analysis, multi-algorithm benchmarking, automated experiment tracking and model registry with **MLflow**, an advanced NIST-compliant optical preprocessing pipeline, and a production-grade **Flask** microservice featuring a bilingual (English / Arabic) corporate web console and RESTful API.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [System Architecture](#system-architecture)
- [Machine Learning & Benchmarking](#machine-learning--benchmarking)
  - [Model Comparison](#model-comparison)
  - [Champion Model: XGBoost](#champion-model-xgboost)
  - [Classification Report & Metrics](#classification-report--metrics)
- [MLOps with MLflow](#mlops-with-mlflow)
- [Optical Preprocessing Pipeline](#optical-preprocessing-pipeline)
- [Web Console & User Interface](#web-console--user-interface)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Quickstart](#installation--quickstart)
  - [Prerequisites](#prerequisites)
  - [Local Environment Setup](#local-environment-setup)
  - [Running the Service](#running-the-service)
- [API Reference](#api-reference)
  - [`GET /health`](#get-health)
  - [`POST /predict`](#post-predict)
- [Docker Deployment](#docker-deployment)
- [Automated Testing](#automated-testing)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Key Highlights

- **98.18% Test Accuracy**: Built and fine-tuned a champion **XGBoost** multiclass model evaluated against SVC, LightGBM, Random Forest, and Logistic Regression.
- **NIST/MNIST Optical Standardization**: Eliminates real-world handwriting distortions using transparency compositing, auto-polarity detection, adaptive noise thresholding, aspect-ratio-preserving 20&times;20 cropping, and sub-pixel **Center-of-Mass (CoM)** alignment.
- **Full MLOps Lifecycle**: Structured tracking via **MLflow** including dynamic experiment versioning, hyperparameter logging, metric collection, artifact saving, and model signature inference.
- **Tri-Modal Input Console**: Accepts input from an **interactive drawing canvas**, **direct file upload** (PNG, JPG, JPEG, WEBP), or **remote image URL**.
- **Bilingual Interface (EN / AR)**: Clean, corporate document verification console designed with zero distracting neon/emojis, complete with real-time telemetry, 28&times;28 preprocessed tensor preview, and full 10-class probability distribution.
- **Production-Ready & Containerized**: Fully containerized using **Docker** (`python:3.9-slim`) with sub-100ms inference latency.

---

## System Architecture

```
[ User Input ]
   ├── Interactive Drawing Canvas (Base64)
   ├── File Upload (PNG / JPG / WEBP)
   └── Remote Image URL (HTTP/HTTPS)
            │
            ▼
[ Optical Preprocessing Engine ]
   ├── 1. Alpha Compositing (Neutral Background)
   ├── 2. Grayscale Conversion ('L')
   ├── 3. Polarity Detection (Auto-Invert Dark-on-Light)
   ├── 4. Adaptive Thresholding & Bounding Box Crop
   ├── 5. Aspect-Preserving Scaling to 20x20
   ├── 6. Center-of-Mass (CoM) Alignment in 28x28 Canvas
   └── 7. Normalization [0.0, 1.0] & 784-Vector Flattening
            │
            ▼
[ XGBoost Classification Engine ] (best_model_xgboost_mnist.pkl)
            │
            ▼
[ Output & Telemetry ]
   ├── Predicted Digit (0–9)
   ├── Confidence Score (%)
   ├── 10-Class Probability Distribution
   ├── 28x28 Preprocessed Tensor Preview (Base64)
   └── Inference Latency (ms)
```

---

## Machine Learning & Benchmarking

The model development process is documented in `Mnist_classification_project_Aya.ipynb`. The dataset was split into 80% training and 20% validation splits (stratified by class labels). Multiple machine learning architectures were trained, evaluated, and compared under identical conditions.

### Model Comparison

| Rank | Model Architecture | Test Accuracy | Status / Notes |
|:---:|:---|:---:|:---|
| 🥇 | **XGBoost Classifier** | **98.18%** | **Selected Champion Model** (Balanced speed, precision, and low false positives) |
| 🥈 | **Support Vector Classifier (SVC)** | **98.18%** | High precision, but slower inference throughput |
| 🥉 | **LightGBM Classifier** | **98.03%** | Fast gradient boosted trees |
| 4 | **Random Forest Classifier** | **96.75%** | 150 Estimators ensemble baseline |
| 5 | **Logistic Regression** | **92.04%** | Linear multiclass baseline (`max_iter=1000`) |

### Champion Model: XGBoost

The champion model utilizes the following hyperparameters for optimal generalization:

```python
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 10,
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1
}
```

### Classification Report & Metrics

```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1343
           1       0.99      0.99      0.99      1600
           2       0.97      0.98      0.97      1380
           3       0.98      0.97      0.97      1433
           4       0.98      0.98      0.98      1295
           5       0.98      0.98      0.98      1273
           6       0.98      0.99      0.99      1349
           7       0.98      0.98      0.98      1422
           8       0.97      0.97      0.97      1397
           9       0.97      0.97      0.97      1408

    accuracy                           0.98     13800
   macro avg       0.98      0.98      0.98     13800
weighted avg       0.98      0.98      0.98     13800
```

---

## MLOps with MLflow

Experiment tracking and model lineage are implemented in `mnist_classification_withMlflow.ipynb`.

- **Tracking URI**: Configurable local or remote MLflow server (`http://localhost:5000`).
- **Experiment Name**: `xgb_Mnist_Prediction_Experiments`.
- **Registered Model**: `MNIST_CLASSIFICATION_Predictor`.
- **Dynamic Versioning**: Automatically inspects prior runs in the MLflow tracking store and generates sequential run version tags (`v1`, `v2`, etc.).
- **Logged Entities**:
  - Dataset source path, row counts, and feature dimensionality.
  - Full hyperparameter set (`n_estimators`, `max_depth`, `learning_rate`, etc.).
  - Validation accuracy, macro-average F1, weighted-average F1, and model footprint (MB).
  - Model artifacts (`joblib_model` serialization) and model signature inference (`infer_signature`).
  - Prediction tracing via `@mlflow.trace`.

To launch the MLflow UI:

```bash
mlflow ui --port 5000
```

---

## Optical Preprocessing Pipeline

Standard MNIST models frequently fail when presented with real-world user drawings due to differences in stroke centering, image dimensions, and background polarity. ODVS includes a custom preprocessing pipeline in `digit_recognition_service/app.py` (`preprocess_image`):

1. **Alpha Channel Compositing**: Merges transparent RGBA/LA canvas exports against a solid white canvas to prevent transparent pixels from becoming noise.
2. **Grayscale Conversion**: Converts arbitrary RGB or paletted inputs into single-channel 8-bit grayscale (`L`).
3. **Background Polarity Auto-Detection**: Samples overall and boundary pixel intensities. If the background is light (mean > 128), it automatically inverts the image so strokes become bright on a dark background (standard MNIST format).
4. **Adaptive Noise Floor & Stroke Isolation**: Isolates stroke ink from paper noise using adaptive percentile thresholding (`max(25.0, p75 * 0.35)`).
5. **Aspect-Ratio Preserving Scale**: Crops the bounding box of the digit and scales it proportionally into a 20&times;20 pixel bounding box using high-quality Lanczos resampling.
6. **Center-of-Mass (CoM) Geometric Centering**:
   - Places the 20&times;20 digit into the center of a 28&times;28 black canvas (4-pixel border padding).
   - Calculates the center of mass $(\bar{x}, \bar{y})$ of the pixel intensities and shifts the image by sub-pixel translation to place the center of mass at the $(13.5, 13.5)$ coordinate.
7. **Normalization & Preview Generation**: Normalizes pixel values to $[0.0, 1.0]$, flattens into 784 named feature columns (`1x1` to `28x28`), and generates an inline Base64 PNG preview for real-time visual inspection in the UI.

---

## Web Console & User Interface

The web interface (`digit_recognition_service/templates/index.html`) is built with a formal, corporate document processing aesthetic:

- **Bilingual Interface**: Full English and Arabic labels for institutional and enterprise workflows.
- **Zero Distraction Design**: Clean slate/navy palette, high-contrast readable typography, and no cartoonish icons or emojis.
- **Live System Telemetry**: Header indicator verifying connection to the Flask server and confirmation of loaded XGBoost weights.
- **Three Intake Modes**:
  1. **Drawing Canvas**: Interactive HTML5 canvas with real-time stroke rendering, adjustable stroke width, and one-click clear.
  2. **File Upload**: Drag-and-drop zone with instant file validation.
  3. **Image URL**: Remote image fetcher with timeout controls.
- **Verification Analysis Report**:
  - Primary predicted digit display.
  - Prediction confidence metric.
  - Complete 10-digit probability distribution bar chart.
  - Preprocessed 28&times;28 optical preview tensor.
  - Detected background shade status and server latency in milliseconds.

---

## Project Directory Structure

```text
Model_Mnist/
├── digit_recognition_service/            # Flask Web Application & Inference Service
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                 # Corporate styling & responsive grid layout
│   │   └── js/
│   │       └── main.js                   # Client-side controller, Canvas logic & AJAX engine
│   ├── templates/
│   │   └── index.html                    # Bilingual (EN/AR) interactive document console
│   ├── app.py                            # Flask application factory, preprocessing & API routes
│   ├── run.py                            # Standalone production service runner script
│   ├── best_model_xgboost_mnist.pkl      # Serialized champion XGBoost model (service copy)
│   ├── test_client.py                    # Automated test client for health & predict endpoints
│   └── requirements.txt                  # Service-specific requirements
├── Mnist/                                # Dataset storage (ignored in Git)
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── Mnist_classification_project_Aya.ipynb # Multi-model exploration, training & evaluation notebook
├── mnist_classification_withMlflow.ipynb  # MLflow tracking, parameter logging & model registration
├── best_model_xgboost_mnist.pkl          # Serialized champion XGBoost model (root copy)
├── requirements.txt                      # Project dependencies
├── Dockerfile                            # Docker containerization configuration
├── .gitignore                            # Git ignore rules (excluding datasets, caches, large artifacts)
└── README.md                             # Comprehensive project documentation
```

---

## Installation & Quickstart

### Prerequisites

- **Python 3.9+**
- **pip** and **virtualenv** (or conda)
- *(Optional)* **Docker** if deploying as a container.

### Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Model_Mnist.git
   cd Model_Mnist
   ```

2. **Create and activate a virtual environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Service

You can start the web service using either `run.py` or `app.py`:

```bash
# Option 1: Standalone Runner (default port 5000)
python digit_recognition_service/run.py

# Option 2: Direct Flask execution (default port 5002)
python digit_recognition_service/app.py
```

Once started, open your web browser and navigate to:
```
http://localhost:5000
```
*(or `http://localhost:5002` if executed via `app.py`)*

---

## API Reference

The service exposes a REST API for automated digit classification.

### `GET /health`

Checks the operational status of the service and confirms model availability.

**Response `200 OK`**:
```json
{
  "status": "ONLINE",
  "model_loaded": true,
  "features_count": 784
}
```

---

### `POST /predict`

Performs optical preprocessing and returns digit classification with probability distributions.

#### Method 1: File Upload (`multipart/form-data`)

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -F "file=@/path/to/digit_image.png"
```

#### Method 2: Interactive Canvas / Base64 (`application/json`)

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"canvas_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}'
```

#### Method 3: Remote Image URL (`application/json`)

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/handwritten_sample.jpg"}'
```

#### Sample JSON Response (`200 OK`)

```json
{
  "success": true,
  "predicted_digit": 8,
  "confidence": 99.42,
  "probabilities": {
    "0": 0.000102,
    "1": 0.000045,
    "2": 0.000215,
    "3": 0.001150,
    "4": 0.000080,
    "5": 0.000312,
    "6": 0.000054,
    "7": 0.000098,
    "8": 0.994210,
    "9": 0.003734
  },
  "distribution": [
    {"digit": 0, "percentage": 0.01, "probability": 0.000102},
    {"digit": 1, "percentage": 0.0, "probability": 0.000045},
    {"digit": 2, "percentage": 0.02, "probability": 0.000215},
    {"digit": 3, "percentage": 0.12, "probability": 0.00115},
    {"digit": 4, "percentage": 0.01, "probability": 0.00008},
    {"digit": 5, "percentage": 0.03, "probability": 0.000312},
    {"digit": 6, "percentage": 0.01, "probability": 0.000054},
    {"digit": 7, "percentage": 0.01, "probability": 0.000098},
    {"digit": 8, "percentage": 99.42, "probability": 0.99421},
    {"digit": 9, "percentage": 0.37, "probability": 0.003734}
  ],
  "background_shade": "Light Media (Inverted to Dark)",
  "inverted": true,
  "pixel_mean": 242.15,
  "preview_image": "data:image/png;base64,iVBORw0KGgo...",
  "input_source": "Interactive Canvas / لوحة رسم تفاعلية",
  "inference_time_ms": 34.12
}
```

---

## Docker Deployment

The application includes a production-ready `Dockerfile` based on `python:3.9-slim`.

1. **Build the Docker container image**:
   ```bash
   docker build -t mnist-odvs-service:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 5000:5000 --name mnist-odvs mnist-odvs-service:latest
   ```

3. **Verify the container**:
   ```bash
   curl http://localhost:5000/health
   ```

---

## Automated Testing

A test suite script is provided in `digit_recognition_service/test_client.py` to validate both health checks and inference routes (simulated canvas rendering and file uploads):

```bash
python digit_recognition_service/test_client.py
```

Expected output:
```text
=== Testing Updated MNIST Verification Service ===
Health Check: 200 {'features_count': 784, 'model_loaded': True, 'status': 'ONLINE'}

--- Canvas Input Result ---
Status: 200
Predicted Digit: 8
Confidence: 99.4 %
Inverted: True
Source: Interactive Canvas / لوحة رسم تفاعلية
Latency: 35.2 ms
```

---

## Author & Acknowledgments

- **Author**: Aya
- **Specialization**: Machine Learning & MLOps Engineering
- **Benchmark Dataset**: [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (Yann LeCun, Corinna Cortes, Christopher J.C. Burges)

