# Heart Disease Prediction - MLOps Assignment

This project implements an end-to-end MLOps pipeline for predicting heart disease using the UCI Heart Disease dataset. It's part of my MLOps coursework where I learned about experiment tracking, CI/CD, containerization, and deployment.

## Overview

The goal was to build a complete ML pipeline that could be deployed to production. I used Logistic Regression and Random Forest models, tracked experiments with MLflow, containerized with Docker, and deployed to Kubernetes. It was quite challenging but I learned a lot about MLOps practices.

## Project Structure

```
.
├── app/
│   └── app.py                 # FastAPI application with /predict endpoint
├── data/
│   └── download_data.py       # Script to download UCI dataset
├── k8s/
│   └── deployment.yaml        # Kubernetes deployment manifest
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
├── src/
│   ├── preprocess.py          # Data cleaning and preprocessing
│   ├── train.py               # Model training with MLflow
│   └── predict.py             # Inference utilities
├── tests/
│   └── test_pipeline.py       # Unit tests
├── .github/workflows/
│   └── ci_cd.yml              # CI/CD pipeline (GitHub Actions)
├── .gitignore                 # Git ignore patterns
├── ARCHITECTURE.md            # System architecture diagram
├── Dockerfile                 # Docker container definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker
- Kubernetes (Minikube, Docker Desktop, or cloud K8s)
- Git

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd MLOPS_ASSIGNMENT

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Data

```bash
python data/download_data.py
```

This downloads the Cleveland Heart Disease dataset from UCI ML Repository. I had some issues initially with the URL but got it working by using the processed.cleveland.data file.

### 2. Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

The EDA notebook shows:
- Data statistics
- Feature distributions
- Class balance
- Correlations
- Some visualization plots

### 3. Train Models

```bash
python src/train.py
```

This trains both Logistic Regression and Random Forest models. I used cross-validation and MLflow for tracking. The best model is selected based on ROC-AUC score and saved for deployment.

### 4. View MLflow UI

```bash
mlflow ui
```

Visit `http://localhost:5000` to see experiment runs, model metrics, and logged artifacts. This was really helpful for comparing different model configurations.

### 5. Run Tests

```bash
pytest tests/test_pipeline.py -v
```

Unit tests cover data preprocessing and prediction output validation. I added these after some issues with feature alignment during inference.

### 6. Run API Locally

```bash
python app/app.py
```

The API will be available at `http://localhost:8000`

**Endpoints:**
- `GET /` - API status
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /predict` - Heart disease prediction

### 7. Sample API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "prediction": 1,
    "label": "Heart Disease Detected",
    "confidence": 0.8234
  },
  "timestamp": "2026-06-29T12:00:00.000000"
}
```

## Containerization

### Build Docker Image

```bash
docker build -t heart-disease-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 heart-disease-api
```

## Kubernetes Deployment

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

### Check Deployment Status

```bash
kubectl get pods
kubectl get services
```

### Access the Service

```bash
# For LoadBalancer service (cloud K8s)
kubectl get svc heart-disease-service

# For Minikube
minikube service heart-disease-service
```

I used a simple deployment with 2 replicas and a LoadBalancer service. The health checks ensure the pod is ready before receiving traffic.

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci_cd.yml`) automatically:
1. Runs linting with flake8
2. Executes unit tests with pytest
3. Trains the model
4. Builds and tests the Docker image

This was my first time setting up a proper CI/CD pipeline - learned a lot about GitHub Actions.

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

I added basic metrics for:
- Total prediction requests (by status)
- Prediction latency
- Heart disease predictions (by class)

### API Logs

Logs are written to `logs/api_requests.log` with request/response data and error messages.

## Model Performance

Based on my training runs:
- **Logistic Regression**: ROC-AUC ~0.85-0.90
- **Random Forest**: ROC-AUC ~0.88-0.92

Random Forest generally performed better, likely due to non-linear relationships in the data. I selected the best model automatically based on ROC-AUC.

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Features**: 14 clinical attributes
- **Target**: Binary classification (0 = No heart disease, 1 = Heart disease)
- **Samples**: 303 patients
- **Missing Values**: Handled via median imputation

The dataset has some missing values represented as '?' which I handled during preprocessing.

## Technology Stack

- **ML Framework**: Scikit-learn
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + custom logging
- **Testing**: Pytest
- **Visualization**: Matplotlib, Seaborn

## Architecture

See `ARCHITECTURE.md` for detailed system architecture diagrams and component descriptions.

## Challenges Faced

- **Feature Alignment**: Had issues with categorical features during inference - fixed by saving and loading feature columns
- **MLflow Setup**: Initially had some configuration issues but got it working
- **Kubernetes Deployment**: Health checks were tricky - had to adjust initial delays
- **CI/CD Debugging**: Pipeline failures due to missing dependencies - fixed by updating requirements.txt

## Future Improvements

- [ ] Add hyperparameter tuning with Optuna
- [ ] Implement model versioning and A/B testing
- [ ] Add input validation for medical data ranges
- [ ] Implement rate limiting and authentication
- [ ] Add more sophisticated feature engineering
- [ ] Deploy monitoring dashboard (Grafana)
- [ ] Add model drift detection

## Academic Integrity

This project was developed as part of MLOps coursework. All code and documentation represent my original work with appropriate use of open-source libraries and frameworks.

## License

This project is for educational purposes.

## Contact

For questions or issues, please open an issue in the repository.