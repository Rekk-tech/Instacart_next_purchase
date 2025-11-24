#!/usr/bin/env python3
"""
Test script to validate dashboard components and data loading
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_data_loading():
    """Test if data files are loading correctly"""
    print("🧪 Testing Data Loading...")
    
    data_dir = current_dir / "data" / "raw"
    
    # Test files existence
    files_to_check = ["orders.csv", "products.csv", "departments.csv", "aisles.csv"]
    for file in files_to_check:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file} exists ({file_path.stat().st_size // (1024*1024)} MB)")
        else:
            print(f"❌ {file} not found")
    
    # Test loading small samples
    try:
        orders = pd.read_csv(data_dir / "orders.csv", nrows=1000)
        print(f"✅ Orders sample loaded: {len(orders)} rows, columns: {list(orders.columns)}")
        
        products = pd.read_csv(data_dir / "products.csv", nrows=100)
        print(f"✅ Products sample loaded: {len(products)} rows, columns: {list(products.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")

def test_model_loading():
    """Test if model can be loaded"""
    print("\n🧪 Testing Model Loading...")
    
    try:
        from src.models.predictor import InstacartPredictor, load_predictor
        
        model_path = current_dir / "experiments" / "checkpoints" / "deployment"
        
        # Check model files
        model_files = ["model.joblib", "standardizer.joblib", "feature_order.json"]
        for file in model_files:
            file_path = model_path / file
            if file_path.exists():
                print(f"✅ {file} exists ({file_path.stat().st_size // 1024} KB)")
            else:
                print(f"❌ {file} not found")
        
        # Try loading model
        predictor = load_predictor(str(model_path))
        print(f"✅ Model loaded successfully")
        
        # Test model info
        model_info = predictor.get_model_info()
        print(f"✅ Model info: {model_info}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def test_sample_prediction():
    """Test making a sample prediction"""
    print("\n🧪 Testing Sample Prediction...")
    
    try:
        from src.models.predictor import load_predictor
        
        model_path = current_dir / "experiments" / "checkpoints" / "deployment"
        predictor = load_predictor(str(model_path))
        
        # Load feature names from config
        import json
        with open(model_path / "feature_order.json", 'r') as f:
            feature_config = json.load(f)
        
        feature_names = feature_config['feature_names']
        print(f"✅ Expected features: {len(feature_names)}")
        print(f"Feature names: {feature_names[:5]}...")
        
        # Create sample features
        sample_features = {}
        for feature in feature_names:
            if 'orders' in feature:
                sample_features[feature] = np.random.randint(1, 50)
            elif 'rate' in feature or 'ratio' in feature:
                sample_features[feature] = np.random.uniform(0, 1)
            elif 'days' in feature:
                sample_features[feature] = np.random.randint(1, 30)
            elif 'hour' in feature:
                sample_features[feature] = np.random.randint(0, 24)
            elif 'dow' in feature:
                sample_features[feature] = np.random.randint(0, 7)
            elif 'id' in feature:
                sample_features[feature] = np.random.randint(1, 100)
            else:
                sample_features[feature] = np.random.uniform(0, 100)
        
        # Make prediction
        features_df = pd.DataFrame([sample_features])
        predictions, probabilities = predictor.predict(features_df, return_proba=True)
        
        print(f"✅ Prediction successful!")
        print(f"Probability: {probabilities[0]:.3f}")
        print(f"Prediction: {predictions[0]}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def test_dashboard_imports():
    """Test dashboard imports"""
    print("\n🧪 Testing Dashboard Imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit available")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly available")
        
        # Test dashboard imports
        sys.path.insert(0, str(current_dir / "reports" / "dashboards"))
        
        print("✅ All dashboard dependencies available")
        
    except Exception as e:
        print(f"❌ Error importing dashboard dependencies: {e}")

def main():
    print("🚀 Instacart Dashboard Validation Test")
    print("=" * 50)
    
    test_data_loading()
    test_model_loading() 
    test_sample_prediction()
    test_dashboard_imports()
    
    print("\n" + "=" * 50)
    print("🎯 Test completed! Check results above.")
    print("If all tests pass ✅, the dashboard should work correctly.")
    print("If tests fail ❌, review the errors and fix dependencies.")

if __name__ == "__main__":
    main()