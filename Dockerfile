# Dockerfile — Allstate Claims Severity API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY app.py .
COPY models/ ./models/

# Set environment variables
ENV MODEL_DIR=./models
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run with gunicorn — 2 workers is appropriate for free-tier hosting
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
