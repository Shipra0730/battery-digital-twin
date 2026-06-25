# Production-grade Dockerfile for Battery PINN Digital Twin Platform
FROM python:3.11-slim

# Prevent python writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for scientific python libraries (scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree and configuration files
COPY src/ ./src/
COPY docker/ ./docker/

EXPOSE 8000
EXPOSE 8501

# Default command can be overridden in docker-compose.yml
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
