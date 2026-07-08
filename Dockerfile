# Dockerfile
# Multi-stage is overkill for this project - keeping it simple
# Initially tried multi-stage but had issues with model files not being copied
# Using single stage for now since the image size isn't too large

FROM python:3.10-slim

# Set working dir
WORKDIR /heart-mlops

# Copy requirements first for layer caching
# This is a Docker best practice - cache dependencies if requirements don't change
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
# Need to include models, src, app, etc.
COPY . .

# Create logs directory
# The API needs this for writing request logs
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Run the API
# Using python directly instead of uvicorn command to keep it simple
CMD ["python", "app/app.py"]