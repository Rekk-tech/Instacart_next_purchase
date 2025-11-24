#!/usr/bin/env python3
"""
Instacart Model CLI Tool
Command-line interface for model training, evaluation, and deployment
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import logging
import sys
import os
from typing import Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models.predictor import InstacartPredictor, load_predictor
from src.utils.logging import setup_logging

# Setup logging
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Instacart Model CLI - Manage ML models and predictions"""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


@cli.group()
def model():
    """Model management commands"""
    pass


@cli.group()
def predict():
    """Prediction commands"""
    pass


@cli.group()
def data():
    """Data processing commands"""
    pass


@model.command('info')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
def model_info(model_dir: Optional[str]):
    """Show model information"""
    try:
        predictor = load_predictor(model_dir)
        info = predictor.get_model_info()
        
        click.echo("\n=== Model Information ===")
        for key, value in info.items():
            click.echo(f"{key}: {value}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@model.command('validate')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
@click.option('--test-data', '-t', type=str, required=True, help='Test data path')
def model_validate(model_dir: Optional[str], test_data: str):
    """Validate model on test data"""
    try:
        # Load model
        predictor = load_predictor(model_dir)
        
        # Load test data
        if test_data.endswith('.parquet'):
            df_test = pd.read_parquet(test_data)
        else:
            df_test = pd.read_csv(test_data)
        
        click.echo(f"Loaded test data: {df_test.shape}")
        
        # Prepare features
        feature_cols = [col for col in df_test.columns if col not in ['user_id', 'product_id', 'reordered']]
        X_test = df_test[feature_cols]
        y_test = df_test['reordered']
        
        # Make predictions
        y_pred, y_pred_proba = predictor.predict(X_test, return_proba=True)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        click.echo("\n=== Validation Results ===")
        for metric, value in metrics.items():
            click.echo(f"{metric}: {value:.4f}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@predict.command('single')
@click.option('--features', '-f', type=str, required=True, help='Features JSON file or JSON string')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
@click.option('--output', '-o', type=str, help='Output file path')
def predict_single(features: str, model_dir: Optional[str], output: Optional[str]):
    """Make single prediction"""
    try:
        # Load model
        predictor = load_predictor(model_dir)
        
        # Parse features
        if Path(features).exists():
            with open(features, 'r') as f:
                feature_dict = json.load(f)
        else:
            feature_dict = json.loads(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction, probability = predictor.predict(features_df, return_proba=True)
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Result saved to: {output}")
        else:
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@predict.command('batch')
@click.option('--input-file', '-i', type=str, required=True, help='Input data file')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
@click.option('--output', '-o', type=str, required=True, help='Output file path')
@click.option('--batch-size', '-b', type=int, default=1000, help='Batch size for processing')
def predict_batch(input_file: str, model_dir: Optional[str], output: str, batch_size: int):
    """Make batch predictions"""
    try:
        # Load model
        predictor = load_predictor(model_dir)
        
        # Load input data
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
        
        click.echo(f"Loaded input data: {df.shape}")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['user_id', 'product_id', 'reordered']]
        X = df[feature_cols]
        
        # Process in batches
        predictions = []
        probabilities = []
        
        n_batches = len(X) // batch_size + (1 if len(X) % batch_size > 0 else 0)
        
        with click.progressbar(range(n_batches), label='Processing batches') as bar:
            for i in bar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                
                X_batch = X.iloc[start_idx:end_idx]
                pred_batch, prob_batch = predictor.predict(X_batch, return_proba=True)
                
                predictions.extend(pred_batch)
                probabilities.extend(prob_batch)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted_reorder'] = predictions
        results_df['reorder_probability'] = probabilities
        
        # Save results
        if output.endswith('.parquet'):
            results_df.to_parquet(output, index=False)
        else:
            results_df.to_csv(output, index=False)
        
        click.echo(f"Results saved to: {output}")
        click.echo(f"Processed {len(results_df):,} predictions")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@predict.command('recommend')
@click.option('--user-id', '-u', type=int, required=True, help='User ID')
@click.option('--candidates', '-c', type=str, required=True, help='Candidate products CSV/parquet file')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
@click.option('--top-k', '-k', type=int, default=20, help='Number of recommendations')
@click.option('--output', '-o', type=str, help='Output file path')
def predict_recommend(user_id: int, candidates: str, model_dir: Optional[str], top_k: int, output: Optional[str]):
    """Generate recommendations for a user"""
    try:
        # Load model
        predictor = load_predictor(model_dir)
        
        # Load candidates
        if candidates.endswith('.parquet'):
            df_candidates = pd.read_parquet(candidates)
        else:
            df_candidates = pd.read_csv(candidates)
        
        # Filter for user
        if 'user_id' in df_candidates.columns:
            df_user = df_candidates[df_candidates['user_id'] == user_id]
        else:
            df_user = df_candidates.copy()
            df_user['user_id'] = user_id
        
        if len(df_user) == 0:
            click.echo(f"No candidates found for user {user_id}")
            return
        
        # Get recommendations
        recommendations = predictor.predict_for_user(df_user, top_k=top_k)
        
        if output:
            if output.endswith('.parquet'):
                recommendations.to_parquet(output, index=False)
            else:
                recommendations.to_csv(output, index=False)
            click.echo(f"Recommendations saved to: {output}")
        else:
            click.echo(recommendations.to_string(index=False))
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@data.command('sample')
@click.option('--input-file', '-i', type=str, required=True, help='Input data file')
@click.option('--output', '-o', type=str, required=True, help='Output file path')
@click.option('--sample-size', '-s', type=int, default=10000, help='Sample size')
@click.option('--random-state', '-r', type=int, default=42, help='Random state for sampling')
def data_sample(input_file: str, output: str, sample_size: int, random_state: int):
    """Sample data from large dataset"""
    try:
        # Load data
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
        
        click.echo(f"Loaded data: {df.shape}")
        
        # Sample data
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=random_state)
            click.echo(f"Sampled {sample_size:,} rows")
        else:
            df_sample = df
            click.echo("Dataset smaller than sample size - using full dataset")
        
        # Save sample
        if output.endswith('.parquet'):
            df_sample.to_parquet(output, index=False)
        else:
            df_sample.to_csv(output, index=False)
        
        click.echo(f"Sample saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command('serve')
@click.option('--host', '-h', default='0.0.0.0', help='Host address')
@click.option('--port', '-p', default=8000, help='Port number')
@click.option('--model-dir', '-d', type=str, help='Model directory path')
def serve(host: str, port: int, model_dir: Optional[str]):
    """Start API server"""
    try:
        import uvicorn
        from src.api.main import app
        
        # Set model directory if provided
        if model_dir:
            import os
            os.environ['MODEL_DIR'] = model_dir
        
        click.echo(f"Starting server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        click.echo("Error: uvicorn not installed. Install with: pip install uvicorn", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == '__main__':
    cli()