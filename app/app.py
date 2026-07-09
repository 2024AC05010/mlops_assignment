import sys
import os
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest

# Add src to path - this was tricky to get right
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import run_prediction  # noqa: E402

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup logging - writing to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics for monitoring
# These help track API performance and usage patterns
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['status']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency'
)
HEART_DISEASE_PREDICTIONS = Counter(
    'heart_disease_predictions_total',
    'Total heart disease predictions',
    ['prediction']
)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Classifier API",
    version="1.0.0"
)


# Request body schema
# Using Pydantic for automatic validation
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def root():
    """Root endpoint - basic API check"""
    return {"message": "Heart Disease Prediction API is running!", "status": "OK"}


@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes liveness probes"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint for monitoring"""
    return generate_latest()


@app.post("/predict")
def predict(patient_data: PatientData):
    """
    Accepts patient health data and returns
    heart disease prediction with confidence score.

    Note: This is a demo/educational API - not for medical use!
    """
    logger.info(f"[REQUEST] /predict called with: {patient_data.dict()}")

    # Track prediction latency using Prometheus histogram
    with PREDICTION_LATENCY.time():
        try:
            input_dict = patient_data.dict()
            result = run_prediction(input_dict)

            # Track metrics for monitoring
            PREDICTION_REQUESTS.labels(status='success').inc()
            HEART_DISEASE_PREDICTIONS.labels(prediction=result['prediction']).inc()

            logger.info(f"[RESPONSE] Prediction: {result}")
            return {
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as err:
            # Track failed requests
            PREDICTION_REQUESTS.labels(status='error').inc()
            logger.error(f"[ERROR] Prediction failed: {str(err)}")
            # Log the full traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(err)}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("STARTING FASTAPI SERVER")
    print("=" * 60)
    print("API will be available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    print("Metrics at: http://localhost:8000/metrics")
    print("=" * 60)

    # Run the server
    # reload=False to avoid issues with module reloading
    uvicorn.run(app, host="0.0.0.0", port=8000)
