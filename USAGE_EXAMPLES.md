# Usage Examples

This document provides comprehensive examples of how to use the Instacart Next Purchase Prediction system.

## Table of Contents

1. [CLI Examples](#cli-examples)
2. [API Examples](#api-examples)
3. [Python SDK Examples](#python-sdk-examples)
4. [Jupyter Notebook Examples](#jupyter-notebook-examples)
5. [Production Integration](#production-integration)

## CLI Examples

### Basic Model Operations

```bash
# Get model information
instacart-cli model-info

# Example output:
# Model Type: Multi-Model Ensemble
# Models: LogisticRegression, XGBoost, LightGBM, TensorFlow
# Version: 1.0.0
# Trained: 2024-01-15 10:30:00
# Features: 25 engineered features
# Accuracy: 0.847

# Validate model integrity
instacart-cli validate-model

# Example output:
# ✓ Model files found
# ✓ Feature schema valid
# ✓ Preprocessing pipeline valid
# ✓ All models loadable
# Model validation passed!
```

### Making Predictions

```bash
# Single prediction with JSON features
instacart-cli predict \
  --user-id 12345 \
  --features '{"days_since_prior_order": 7, "order_hour_of_day": 14, "order_dow": 1, "user_total_orders": 25, "user_reorder_ratio": 0.75}'

# Example output:
# User ID: 12345
# Reorder Probability: 0.734
# Confidence: High
# Top Products: [Banana, Strawberries, Yogurt]

# Batch predictions from CSV
instacart-cli batch-predict \
  --input data/sample_features.csv \
  --output results/predictions.csv \
  --batch-size 1000

# Example output:
# Processing 10,000 predictions...
# ████████████████████████████████ 100%
# Results saved to results/predictions.csv
# Average probability: 0.652
# High confidence predictions: 7,843 (78.4%)
```

### Data Operations

```bash
# Generate sample data for testing
instacart-cli sample-data \
  --size 100 \
  --output sample_users.csv \
  --include-labels

# Example output:
# Generated 100 sample records
# Features: days_since_prior_order, order_hour_of_day, order_dow, etc.
# Labels: reordered (for validation)
# Saved to sample_users.csv

# Start API server
instacart-cli serve \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# Example output:
# Starting Instacart Prediction API...
# Model loaded successfully
# Server running at http://0.0.0.0:8000
# Workers: 4
# Swagger UI: http://0.0.0.0:8000/docs
```

## API Examples

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Response:
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "uptime": 3600
}

# Readiness check (includes model status)
curl http://localhost:8000/ready

# Response:
{
    "status": "ready",
    "model_loaded": true,
    "model_version": "1.0.0",
    "last_prediction": "2024-01-15T10:29:45Z",
    "predictions_served": 1234
}
```

### Single Predictions

```bash
# Make single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "features": {
        "days_since_prior_order": 7,
        "order_hour_of_day": 14,
        "order_dow": 1,
        "user_total_orders": 25,
        "user_reorder_ratio": 0.75,
        "user_avg_days_between_orders": 8.5,
        "user_total_products": 156,
        "user_total_distinct_products": 89,
        "department_0": 0.15,
        "department_1": 0.12
    }
}'

# Response:
{
    "user_id": 12345,
    "prediction": {
        "reorder_probability": 0.734,
        "confidence": "high",
        "model_ensemble": {
            "logistic_regression": 0.729,
            "xgboost": 0.741,
            "lightgbm": 0.738,
            "tensorflow": 0.732
        }
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time_ms": 45
}
```

### Batch Predictions

```bash
# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
        {
            "user_id": 12345,
            "features": {
                "days_since_prior_order": 7,
                "order_hour_of_day": 14,
                "order_dow": 1,
                "user_total_orders": 25,
                "user_reorder_ratio": 0.75
            }
        },
        {
            "user_id": 12346,
            "features": {
                "days_since_prior_order": 14,
                "order_hour_of_day": 10,
                "order_dow": 6,
                "user_total_orders": 12,
                "user_reorder_ratio": 0.65
            }
        }
    ]
}'

# Response:
{
    "predictions": [
        {
            "user_id": 12345,
            "reorder_probability": 0.734,
            "confidence": "high"
        },
        {
            "user_id": 12346,
            "reorder_probability": 0.612,
            "confidence": "medium"
        }
    ],
    "batch_stats": {
        "total_predictions": 2,
        "average_probability": 0.673,
        "processing_time_ms": 23
    }
}
```

### Recommendations

```bash
# Get product recommendations for user
curl -X POST http://localhost:8000/recommend/12345 \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 10,
    "include_metadata": true,
    "filter_departments": [4, 16, 7]
}'

# Response:
{
    "user_id": 12345,
    "recommendations": [
        {
            "product_id": 24852,
            "product_name": "Banana",
            "department": "produce",
            "aisle": "fresh fruits",
            "reorder_probability": 0.892,
            "rank": 1
        },
        {
            "product_id": 13176,
            "product_name": "Bag of Organic Bananas",
            "department": "produce", 
            "aisle": "fresh fruits",
            "reorder_probability": 0.756,
            "rank": 2
        }
    ],
    "metadata": {
        "model_version": "1.0.0",
        "recommendation_strategy": "collaborative_filtering",
        "user_segment": "frequent_shopper"
    }
}
```

### Model Management

```bash
# Get model information
curl http://localhost:8000/model/info

# Response:
{
    "model_info": {
        "version": "1.0.0",
        "type": "ensemble",
        "models": ["logistic_regression", "xgboost", "lightgbm", "tensorflow"],
        "features": 25,
        "training_date": "2024-01-15",
        "accuracy_metrics": {
            "precision": 0.823,
            "recall": 0.867,
            "f1_score": 0.845,
            "roc_auc": 0.912
        }
    }
}

# Get model performance metrics
curl http://localhost:8000/model/metrics

# Response:
{
    "performance_metrics": {
        "total_predictions": 15678,
        "avg_response_time_ms": 42,
        "prediction_accuracy": 0.847,
        "last_24h_predictions": 2341,
        "error_rate": 0.002
    },
    "system_metrics": {
        "memory_usage_mb": 2048,
        "cpu_usage_percent": 15.4,
        "uptime_hours": 72.5
    }
}

# Reload model (for updates)
curl -X POST http://localhost:8000/model/reload

# Response:
{
    "status": "success",
    "message": "Model reloaded successfully",
    "new_version": "1.0.1",
    "reload_time_ms": 1250
}
```

## Python SDK Examples

### Basic Usage

```python
from src.models.predictor import InstacartPredictor
import pandas as pd

# Load the predictor
predictor = InstacartPredictor.load_predictor("models/")

# Single prediction
features = {
    'days_since_prior_order': 7,
    'order_hour_of_day': 14,
    'order_dow': 1,
    'user_total_orders': 25,
    'user_reorder_ratio': 0.75,
    'user_avg_days_between_orders': 8.5
}

probability = predictor.predict(features)
print(f"Reorder probability: {probability:.3f}")

# Batch predictions
df = pd.read_csv('sample_features.csv')
probabilities = predictor.predict_batch(df)
print(f"Average probability: {probabilities.mean():.3f}")
```

### Advanced Usage with Validation

```python
import numpy as np
from src.models.predictor import InstacartPredictor, load_predictor

# Load predictor with error handling
try:
    predictor = load_predictor("models/")
    print(f"Model loaded: {predictor.model_version}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Validate features before prediction
features = {
    'days_since_prior_order': 7,
    'order_hour_of_day': 14,
    'order_dow': 1,
    'user_total_orders': 25,
    'user_reorder_ratio': 0.75
}

if predictor.validate_features(features):
    probability = predictor.predict(features)
    confidence = predictor.get_prediction_confidence(probability)
    
    print(f"Prediction: {probability:.3f}")
    print(f"Confidence: {confidence}")
    
    # Get ensemble breakdown
    if hasattr(predictor, 'predict_ensemble'):
        ensemble_predictions = predictor.predict_ensemble(features)
        print("Ensemble predictions:")
        for model, pred in ensemble_predictions.items():
            print(f"  {model}: {pred:.3f}")
else:
    print("Invalid features provided")
```

### Streaming Predictions

```python
import asyncio
from src.models.predictor import InstacartPredictor

async def stream_predictions(feature_generator):
    """Process predictions as they come in"""
    predictor = InstacartPredictor.load_predictor("models/")
    
    async for features in feature_generator:
        try:
            probability = predictor.predict(features)
            yield {
                'user_id': features.get('user_id'),
                'probability': probability,
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            yield {'error': str(e), 'features': features}

# Usage
async def main():
    async for result in stream_predictions(feature_stream):
        if 'error' not in result:
            print(f"User {result['user_id']}: {result['probability']:.3f}")
        else:
            print(f"Error: {result['error']}")

# Run
asyncio.run(main())
```

## Jupyter Notebook Examples

### Interactive Exploration

```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.predictor import InstacartPredictor

# Load model
predictor = InstacartPredictor.load_predictor("models/")
print(f"Model loaded: {predictor.model_version}")

# Cell 2: Sample Data Analysis
# Load sample data
df = pd.read_csv('data/sample_features.csv')
print(f"Sample size: {len(df)}")
print(df.head())

# Cell 3: Make Predictions
predictions = predictor.predict_batch(df)
df['prediction'] = predictions

# Cell 4: Visualize Results
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Reorder Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Reorder Probabilities')
plt.axvline(predictions.mean(), color='red', linestyle='--', 
           label=f'Mean: {predictions.mean():.3f}')
plt.legend()
plt.show()

# Cell 5: Feature Analysis
feature_importance = predictor.get_feature_importance()
plt.figure(figsize=(12, 8))
feature_importance.plot(kind='barh')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

### Model Comparison

```python
# Cell 1: Load different model versions
models = {
    'v1.0': InstacartPredictor.load_predictor("models/v1.0/"),
    'v1.1': InstacartPredictor.load_predictor("models/v1.1/"),
    'latest': InstacartPredictor.load_predictor("models/")
}

# Cell 2: Compare predictions
test_features = df.sample(1000)
results = {}

for name, model in models.items():
    preds = model.predict_batch(test_features)
    results[name] = preds

# Cell 3: Comparison visualization
comparison_df = pd.DataFrame(results)
correlation_matrix = comparison_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Model Prediction Correlations')
plt.show()

# Cell 4: Performance metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve

if 'actual' in test_features.columns:
    for name, preds in results.items():
        auc = roc_auc_score(test_features['actual'], preds)
        print(f"{name} AUC: {auc:.3f}")
```

## Production Integration

### FastAPI Integration

```python
# custom_api.py
from fastapi import FastAPI, BackgroundTasks
from src.models.predictor import InstacartPredictor
import asyncio

app = FastAPI(title="Custom Instacart API")

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = InstacartPredictor.load_predictor("models/")

@app.post("/custom-predict")
async def custom_predict(features: dict, background_tasks: BackgroundTasks):
    """Custom prediction endpoint with logging"""
    
    # Make prediction
    probability = predictor.predict(features)
    
    # Log prediction in background
    background_tasks.add_task(log_prediction, features, probability)
    
    return {
        "probability": probability,
        "recommendation": "reorder" if probability > 0.5 else "skip",
        "confidence": predictor.get_prediction_confidence(probability)
    }

async def log_prediction(features: dict, probability: float):
    """Log prediction for monitoring"""
    # Implement your logging logic here
    pass
```

### Celery Integration (Async Processing)

```python
# tasks.py
from celery import Celery
from src.models.predictor import InstacartPredictor
import pandas as pd

app = Celery('instacart_predictions')

# Load predictor once
predictor = InstacartPredictor.load_predictor("models/")

@app.task
def batch_predict_task(csv_path: str, output_path: str):
    """Process batch predictions asynchronously"""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Make predictions
    predictions = predictor.predict_batch(df)
    
    # Save results
    df['reorder_probability'] = predictions
    df.to_csv(output_path, index=False)
    
    return {
        'status': 'completed',
        'processed_records': len(df),
        'average_probability': predictions.mean(),
        'output_file': output_path
    }

@app.task
def model_health_check():
    """Periodic model health check"""
    try:
        # Test prediction
        test_features = {
            'days_since_prior_order': 7,
            'order_hour_of_day': 14,
            'order_dow': 1,
            'user_total_orders': 25,
            'user_reorder_ratio': 0.75
        }
        
        probability = predictor.predict(test_features)
        
        return {
            'status': 'healthy',
            'test_probability': probability,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }

# Usage
if __name__ == "__main__":
    # Start batch job
    result = batch_predict_task.delay('input.csv', 'output.csv')
    print(f"Task ID: {result.id}")
    
    # Check health
    health = model_health_check.delay()
    print(health.get())
```

### Apache Kafka Integration (Real-time)

```python
# kafka_consumer.py
from kafka import KafkaConsumer, KafkaProducer
import json
from src.models.predictor import InstacartPredictor

# Initialize
predictor = InstacartPredictor.load_predictor("models/")
consumer = KafkaConsumer(
    'user_features',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Process messages
for message in consumer:
    try:
        # Extract features
        user_data = message.value
        features = user_data.get('features', {})
        user_id = user_data.get('user_id')
        
        # Make prediction
        probability = predictor.predict(features)
        
        # Send result
        result = {
            'user_id': user_id,
            'probability': probability,
            'timestamp': pd.Timestamp.now().isoformat(),
            'confidence': predictor.get_prediction_confidence(probability)
        }
        
        producer.send('predictions', value=result)
        print(f"Processed user {user_id}: {probability:.3f}")
        
    except Exception as e:
        print(f"Error processing message: {e}")
        # Send error message
        error_result = {
            'user_id': user_data.get('user_id', 'unknown'),
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        producer.send('prediction_errors', value=error_result)
```

### Database Integration

```python
# database_integration.py
import sqlite3
import pandas as pd
from src.models.predictor import InstacartPredictor
from datetime import datetime

class PredictionDatabase:
    def __init__(self, db_path: str, model_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.predictor = InstacartPredictor.load_predictor(model_path)
        self.init_tables()
    
    def init_tables(self):
        """Initialize database tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                probability REAL NOT NULL,
                confidence TEXT,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS batch_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total_records INTEGER,
                completed_records INTEGER,
                start_time DATETIME,
                end_time DATETIME
            )
        ''')
        self.conn.commit()
    
    def predict_and_store(self, user_id: int, features: dict):
        """Make prediction and store in database"""
        probability = self.predictor.predict(features)
        confidence = self.predictor.get_prediction_confidence(probability)
        
        self.conn.execute('''
            INSERT INTO predictions 
            (user_id, probability, confidence, features, model_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id, 
            probability, 
            confidence,
            json.dumps(features),
            self.predictor.model_version
        ))
        self.conn.commit()
        
        return probability
    
    def get_user_history(self, user_id: int, limit: int = 10):
        """Get prediction history for user"""
        df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', self.conn, params=(user_id, limit))
        
        return df
    
    def batch_predict_from_db(self, query: str, job_id: str):
        """Process batch predictions from database query"""
        
        # Start job
        self.conn.execute('''
            INSERT INTO batch_jobs (job_id, status, start_time)
            VALUES (?, 'running', ?)
        ''', (job_id, datetime.now()))
        self.conn.commit()
        
        try:
            # Load data
            df = pd.read_sql_query(query, self.conn)
            total_records = len(df)
            
            # Update job
            self.conn.execute('''
                UPDATE batch_jobs 
                SET total_records = ? 
                WHERE job_id = ?
            ''', (total_records, job_id))
            
            # Process predictions
            predictions = []
            for idx, row in df.iterrows():
                features = row.drop(['user_id']).to_dict()
                probability = self.predictor.predict(features)
                
                predictions.append({
                    'user_id': row['user_id'],
                    'probability': probability,
                    'confidence': self.predictor.get_prediction_confidence(probability),
                    'features': json.dumps(features),
                    'model_version': self.predictor.model_version
                })
                
                # Update progress every 100 records
                if idx % 100 == 0:
                    self.conn.execute('''
                        UPDATE batch_jobs 
                        SET completed_records = ? 
                        WHERE job_id = ?
                    ''', (idx + 1, job_id))
                    self.conn.commit()
            
            # Bulk insert predictions
            pred_df = pd.DataFrame(predictions)
            pred_df.to_sql('predictions', self.conn, if_exists='append', index=False)
            
            # Complete job
            self.conn.execute('''
                UPDATE batch_jobs 
                SET status = 'completed', end_time = ?, completed_records = ?
                WHERE job_id = ?
            ''', (datetime.now(), total_records, job_id))
            self.conn.commit()
            
            return len(predictions)
            
        except Exception as e:
            # Fail job
            self.conn.execute('''
                UPDATE batch_jobs 
                SET status = 'failed', end_time = ?
                WHERE job_id = ?
            ''', (datetime.now(), job_id))
            self.conn.commit()
            raise e

# Usage example
if __name__ == "__main__":
    db = PredictionDatabase("predictions.db", "models/")
    
    # Single prediction
    features = {'days_since_prior_order': 7, 'order_hour_of_day': 14}
    probability = db.predict_and_store(12345, features)
    print(f"Prediction stored: {probability:.3f}")
    
    # Batch job
    job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    query = "SELECT user_id, days_since_prior_order, order_hour_of_day FROM user_features LIMIT 1000"
    count = db.batch_predict_from_db(query, job_id)
    print(f"Processed {count} predictions in job {job_id}")
```

This comprehensive usage guide covers all the major ways to interact with the Instacart prediction system, from simple CLI commands to complex production integrations.