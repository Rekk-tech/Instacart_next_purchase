# Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the Instacart Next Purchase Prediction system in various environments.

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/instacart-next-purchase
cd instacart-next-purchase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# Or install with all extras
pip install -e ".[all]"
```

### 2. Data Preparation

```bash
# Download Instacart dataset
# Place CSV files in data/raw/

# Run ETL pipeline
python -m src.etl.extract
python -m src.etl.transform
```

### 3. Model Training

```bash
# Train models using notebook
jupyter notebook notebooks/03_Modeling_Experiments_Optimized.ipynb

# Or train via CLI
instacart-cli train --config src/config/default.yaml
```

### 4. Run the System

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

Services available:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Nginx**: http://localhost:80
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

#### Option B: Manual Start

```bash
# Start API server
instacart-cli serve --port 8000

# Or directly with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## CLI Usage

### Model Operations

```bash
# Get model information
instacart-cli model-info

# Validate model
instacart-cli validate-model

# Make predictions
instacart-cli predict --user-id 123 --features user_features.json

# Batch predictions
instacart-cli batch-predict --input predictions_input.csv --output results.csv

# Generate sample data
instacart-cli sample-data --size 100 --output sample.csv
```

### API Operations

```bash
# Start API server
instacart-cli serve --host 0.0.0.0 --port 8000

# Test API health
curl http://localhost:8000/health
```

## API Endpoints

### Health Check
```
GET /health
GET /ready
```

### Predictions
```
POST /predict
Content-Type: application/json
{
    "user_id": 123,
    "features": {...}
}

POST /predict/batch
Content-Type: application/json
{
    "predictions": [
        {"user_id": 123, "features": {...}},
        {"user_id": 456, "features": {...}}
    ]
}
```

### Recommendations
```
POST /recommend/{user_id}
Content-Type: application/json
{
    "top_k": 10,
    "include_metadata": true
}
```

### Model Management
```
GET /model/info
GET /model/metrics
POST /model/reload
```

## Production Deployment

### Docker Production

```bash
# Build production image
docker build -t instacart-predictor:prod .

# Run with production settings
docker run -d \
  --name instacart-api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -v $(pwd)/models:/app/models \
  instacart-predictor:prod
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: instacart-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: instacart-api
  template:
    metadata:
      labels:
        app: instacart-api
    spec:
      containers:
      - name: api
        image: instacart-predictor:prod
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: instacart-service
spec:
  selector:
    app: instacart-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Cloud Deployment (AWS)

```bash
# Using AWS ECS/Fargate
aws ecs create-cluster --cluster-name instacart-cluster

# Build and push to ECR
aws ecr create-repository --repository-name instacart-predictor
docker tag instacart-predictor:prod ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/instacart-predictor:latest
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/instacart-predictor:latest
```

## Monitoring and Logging

### Prometheus Metrics

Available at `/metrics` endpoint:
- Request count and latency
- Model prediction accuracy
- System resources (CPU, memory)
- Custom business metrics

### Grafana Dashboards

Import dashboards from `infra/monitoring/dashboards/`:
- API Performance Dashboard
- Model Performance Dashboard  
- System Health Dashboard

### Logging

Logs are structured JSON format with levels:
- INFO: Normal operations
- WARNING: Performance issues
- ERROR: Application errors
- DEBUG: Detailed debugging info

## Environment Variables

```bash
# Core settings
ENVIRONMENT=development|staging|production
LOG_LEVEL=INFO
MODEL_PATH=/app/models

# API settings
API_HOST=0.0.0.0
API_PORT=8000
MAX_WORKERS=4

# Database (if applicable)
DATABASE_URL=postgresql://user:pass@host:5432/instacart

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
API_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

## Performance Tuning

### API Optimization

```python
# src/api/main.py
app = FastAPI(
    title="Instacart Predictor",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware for performance
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Model Optimization

```python
# Batch prediction optimization
predictor = InstacartPredictor.load_predictor("models/")
predictions = predictor.predict_batch(features_df, batch_size=1000)
```

## Security Considerations

### API Security

```python
# Add API key authentication
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate API key
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(401, "Invalid API key")
```

### Container Security

```dockerfile
# Use non-root user
RUN groupadd -r instacart && useradd -r -g instacart instacart
USER instacart

# Scan for vulnerabilities
RUN pip install safety
RUN safety check
```

## Troubleshooting

### Common Issues

1. **Model loading errors**
   ```bash
   # Check model files exist
   ls -la models/
   
   # Validate model format
   instacart-cli validate-model
   ```

2. **API startup failures**
   ```bash
   # Check logs
   docker-compose logs api
   
   # Test health endpoint
   curl http://localhost:8000/health
   ```

3. **Performance issues**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check Prometheus metrics
   curl http://localhost:8000/metrics
   ```

### Debug Mode

```bash
# Run in debug mode
export LOG_LEVEL=DEBUG
instacart-cli serve --debug

# Or with Docker
docker-compose -f docker-compose.debug.yml up
```

## Scaling

### Horizontal Scaling

```bash
# Scale API replicas
docker-compose up --scale api=3

# Or with Kubernetes
kubectl scale deployment instacart-api --replicas=5
```

### Load Balancing

Configure nginx upstream:
```nginx
upstream instacart_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}
```

## Maintenance

### Model Updates

```bash
# Deploy new model
instacart-cli deploy-model --model-path new_model/

# Reload without downtime
curl -X POST http://localhost:8000/model/reload
```

### Database Migrations

```bash
# Run migrations
alembic upgrade head

# Or with Docker
docker-compose run api alembic upgrade head
```

## Support

For issues and questions:
- Check logs first: `docker-compose logs`
- Review metrics: http://localhost:9090
- Open GitHub issue with logs and error details