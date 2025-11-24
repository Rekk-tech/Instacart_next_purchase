"""
Test suite for Instacart predictor
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.predictor import InstacartPredictor, load_predictor, quick_predict


class TestInstacartPredictor:
    """Test cases for InstacartPredictor class"""
    
    @pytest.fixture
    def sample_features(self):
        """Sample feature data"""
        return pd.DataFrame({
            'user_total_orders': [10, 15, 8],
            'user_avg_days_between_orders': [7.5, 10.2, 5.1],
            'product_total_orders': [1000, 500, 800],
            'product_reorder_rate': [0.3, 0.2, 0.4],
            'user_product_orders': [2, 1, 3],
            'days_since_last_order': [5, 8, 3],
            'hour_of_day': [14, 10, 18],
            'day_of_week': [1, 5, 2],
            'add_to_cart_order': [1, 2, 1],
            'aisle_id': [24, 15, 30],
            'department_id': [4, 2, 6]
        })
    
    @pytest.fixture
    def mock_model_artifacts(self, tmp_path, sample_features):
        """Create mock model artifacts"""
        # Create mock model
        X_sample = sample_features
        y_sample = np.array([1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_sample, y_sample)
        
        # Create mock scaler
        scaler = StandardScaler()
        scaler.fit(X_sample)
        
        # Create mock config
        config = {
            'feature_names': list(sample_features.columns),
            'n_features': len(sample_features.columns),
            'preprocessing': {'scaling_required': True}
        }
        
        # Save artifacts
        model_path = tmp_path / "model.joblib"
        scaler_path = tmp_path / "standardizer.joblib"
        config_path = tmp_path / "feature_order.json"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'config_path': str(config_path),
            'model_dir': str(tmp_path)
        }
    
    def test_predictor_initialization(self, mock_model_artifacts):
        """Test predictor initialization"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        assert predictor.is_loaded
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.feature_config is not None
    
    def test_feature_validation(self, mock_model_artifacts, sample_features):
        """Test feature validation"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        # Valid features
        validated_features = predictor.validate_features(sample_features)
        assert len(validated_features) == len(sample_features)
        assert list(validated_features.columns) == list(sample_features.columns)
        
        # Missing features
        incomplete_features = sample_features.drop('user_total_orders', axis=1)
        with pytest.raises(ValueError):
            predictor.validate_features(incomplete_features)
    
    def test_prediction(self, mock_model_artifacts, sample_features):
        """Test basic prediction functionality"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        # Test prediction with probabilities
        predictions, probabilities = predictor.predict(sample_features, return_proba=True)
        
        assert len(predictions) == len(sample_features)
        assert len(probabilities) == len(sample_features)
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)
        
        # Test prediction without probabilities
        predictions_only = predictor.predict(sample_features, return_proba=False)
        assert len(predictions_only) == len(sample_features)
    
    def test_user_recommendations(self, mock_model_artifacts, sample_features):
        """Test user recommendation functionality"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        # Add user_id and product_id columns
        user_features = sample_features.copy()
        user_features['user_id'] = 1
        user_features['product_id'] = [101, 102, 103]
        
        recommendations = predictor.predict_for_user(user_features, top_k=2)
        
        assert len(recommendations) <= 2
        assert 'user_id' in recommendations.columns
        assert 'product_id' in recommendations.columns
        assert 'reorder_probability' in recommendations.columns
        assert recommendations['reorder_probability'].is_monotonic_decreasing
    
    def test_batch_prediction(self, mock_model_artifacts, sample_features):
        """Test batch prediction"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        batch = [sample_features.iloc[:1], sample_features.iloc[1:2], sample_features.iloc[2:]]
        predictions_batch = predictor.batch_predict(batch)
        
        assert len(predictions_batch) == 3
        assert all(len(pred) == 1 for pred in predictions_batch)
    
    def test_model_info(self, mock_model_artifacts):
        """Test model info retrieval"""
        predictor = InstacartPredictor(
            model_path=mock_model_artifacts['model_path'],
            scaler_path=mock_model_artifacts['scaler_path'],
            config_path=mock_model_artifacts['config_path']
        )
        
        info = predictor.get_model_info()
        
        assert 'model_type' in info
        assert 'model_loaded' in info
        assert info['model_loaded'] is True
        assert info['scaler_available'] is True
        assert info['feature_config_available'] is True
    
    def test_load_predictor_function(self, mock_model_artifacts):
        """Test load_predictor convenience function"""
        predictor = load_predictor(mock_model_artifacts['model_dir'])
        
        assert predictor.is_loaded
        assert predictor.model is not None
    
    def test_quick_predict_function(self, mock_model_artifacts, sample_features):
        """Test quick_predict convenience function"""
        predictions, probabilities = quick_predict(
            sample_features, 
            mock_model_artifacts['model_dir']
        )
        
        assert len(predictions) == len(sample_features)
        assert len(probabilities) == len(sample_features)
    
    def test_error_handling(self):
        """Test error handling for missing artifacts"""
        with pytest.raises(FileNotFoundError):
            InstacartPredictor(model_path="nonexistent.joblib")
    
    def test_predictor_without_scaler(self, tmp_path, sample_features):
        """Test predictor functionality without scaler"""
        # Create model without scaler
        X_sample = sample_features
        y_sample = np.array([1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_sample, y_sample)
        
        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)
        
        predictor = InstacartPredictor(
            model_path=str(model_path),
            scaler_path="nonexistent.joblib",  # This should not exist
            config_path="nonexistent.json"     # This should not exist
        )
        
        # Should still work without scaler
        predictions = predictor.predict(sample_features, return_proba=False)
        assert len(predictions) == len(sample_features)


class TestAPIIntegration:
    """Test API integration (requires running server)"""
    
    @pytest.mark.integration
    def test_api_health_endpoint(self):
        """Test API health endpoint"""
        import requests
        
        try:
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200
            data = response.json()
            assert 'status' in data
            assert 'model_loaded' in data
        except requests.ConnectionError:
            pytest.skip("API server not running")
    
    @pytest.mark.integration
    def test_api_prediction_endpoint(self):
        """Test API prediction endpoint"""
        import requests
        
        try:
            features = {
                'user_total_orders': 10,
                'user_avg_days_between_orders': 7.5,
                'product_total_orders': 1000,
                'product_reorder_rate': 0.3,
                'user_product_orders': 2,
                'days_since_last_order': 5,
                'hour_of_day': 14,
                'day_of_week': 1,
                'add_to_cart_order': 1,
                'aisle_id': 24,
                'department_id': 4
            }
            
            payload = {'features': features}
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert 'prediction' in data
            assert 'probability' in data
        except requests.ConnectionError:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])