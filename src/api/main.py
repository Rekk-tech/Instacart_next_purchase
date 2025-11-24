"""
FastAPI service for Instacart reorder prediction
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
from pathlib import Path

from src.models.predictor import InstacartPredictor, load_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Instacart Reorder Prediction API",
    description="API for predicting product reorders using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[InstacartPredictor] = None


# Pydantic models for API
class FeatureSet(BaseModel):
    """Single feature set for prediction"""
    user_id: int
    product_id: int
    features: Dict[str, float] = Field(..., description="Feature values as key-value pairs")


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    features: Dict[str, float] = Field(..., description="Feature values")
    user_id: Optional[int] = Field(None, description="User ID")
    product_id: Optional[int] = Field(None, description="Product ID")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    feature_sets: List[FeatureSet] = Field(..., description="List of feature sets")


class RecommendationRequest(BaseModel):
    """Request model for user recommendations"""
    user_id: int
    product_features: List[Dict[str, float]] = Field(..., description="Features for candidate products")
    product_ids: Optional[List[int]] = Field(None, description="Product IDs corresponding to features")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    prediction: int
    probability: float
    user_id: Optional[int] = None
    product_id: Optional[int] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int
    recommendations: List[Dict[str, float]]
    total_candidates: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Dict
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor
    try:
        predictor = load_predictor()
        logger.info("Predictor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        predictor = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = predictor.get_model_info() if predictor else {}
    
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        predictions, probabilities = predictor.predict(features_df, return_proba=True)
        
        return PredictionResponse(
            prediction=int(predictions[0]),
            probability=float(probabilities[0]),
            user_id=request.user_id,
            product_id=request.product_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions_list = []
        
        for feature_set in request.feature_sets:
            # Convert features to DataFrame
            features_df = pd.DataFrame([feature_set.features])
            
            # Make prediction
            predictions, probabilities = predictor.predict(features_df, return_proba=True)
            
            predictions_list.append(PredictionResponse(
                prediction=int(predictions[0]),
                probability=float(probabilities[0]),
                user_id=feature_set.user_id,
                product_id=feature_set.product_id,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total_processed=len(predictions_list),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get product recommendations for a user"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features DataFrame
        features_data = []
        for i, product_features in enumerate(request.product_features):
            features_data.append({
                'user_id': request.user_id,
                'product_id': request.product_ids[i] if request.product_ids else i,
                **product_features
            })
        
        features_df = pd.DataFrame(features_data)
        
        # Get recommendations
        recommendations_df = predictor.predict_for_user(features_df, top_k=request.top_k)
        
        # Convert to response format
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append({
                'product_id': int(row['product_id']),
                'reorder_probability': float(row['reorder_probability']),
                'reorder_prediction': int(row['reorder_prediction'])
            })
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_candidates=len(request.product_features),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()


# Reload model endpoint
@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload model (for model updates)"""
    def _reload_model():
        global predictor
        try:
            predictor = load_predictor()
            logger.info("Model reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
    
    background_tasks.add_task(_reload_model)
    return {"message": "Model reload initiated"}


# Example endpoints for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Instacart Reorder Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Health check",
            "/predict - Single prediction", 
            "/predict/batch - Batch predictions",
            "/recommend - Get recommendations",
            "/model/info - Model information",
            "/docs - API documentation"
        ]
    }


@app.get("/example/features")
async def example_features():
    """Example feature structure"""
    return {
        "example_features": {
            "user_total_orders": 10,
            "user_avg_days_between_orders": 7.5,
            "product_total_orders": 1000,
            "product_reorder_rate": 0.3,
            "user_product_orders": 2,
            "days_since_last_order": 5,
            "hour_of_day": 14,
            "day_of_week": 1,
            "add_to_cart_order": 1,
            "aisle_id": 24,
            "department_id": 4
        },
        "required_fields": "Check /model/info for exact feature names"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)