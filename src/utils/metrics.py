"""
Metrics utilities for model evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, log_loss, accuracy_score
)
from typing import Dict, List, Optional, Tuple, Union
import warnings
from .logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Calculate precision@k for recommendation systems."""
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Get top-k predictions
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_predictions = np.zeros_like(y_true)
    top_k_predictions[top_k_indices] = 1
    
    # Calculate precision@k
    if np.sum(top_k_predictions) == 0:
        return 0.0
    
    return np.sum(y_true[top_k_indices]) / k


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Calculate recall@k for recommendation systems."""
    if np.sum(y_true) == 0:
        return 0.0
    
    # Get top-k predictions
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # Calculate recall@k
    return np.sum(y_true[top_k_indices]) / np.sum(y_true)


def f1_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Calculate F1@k for recommendation systems."""
    precision_k = precision_at_k(y_true, y_scores, k)
    recall_k = recall_at_k(y_true, y_scores, k)
    
    if precision_k + recall_k == 0:
        return 0.0
    
    return 2 * (precision_k * recall_k) / (precision_k + recall_k)


def mean_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate Mean Average Precision (MAP)."""
    return average_precision_score(y_true, y_scores)


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at k."""
    # Get top-k indices
    top_k_indices = np.argsort(y_scores)[-k:][::-1]  # Reverse for descending order
    
    # Calculate DCG@k
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        dcg += y_true[idx] / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG@k (Ideal DCG)
    ideal_indices = np.argsort(y_true)[-k:][::-1]
    idcg = 0.0
    for i, idx in enumerate(ideal_indices):
        idcg += y_true[idx] / np.log2(i + 2)
    
    # Return NDCG@k
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Calculate hit rate@k (whether any relevant item is in top-k)."""
    top_k_indices = np.argsort(y_scores)[-k:]
    return 1.0 if np.sum(y_true[top_k_indices]) > 0 else 0.0


def coverage_at_k(y_scores_list: List[np.ndarray], k: int, total_items: int) -> float:
    """Calculate catalog coverage@k across all users."""
    recommended_items = set()
    
    for y_scores in y_scores_list:
        top_k_indices = np.argsort(y_scores)[-k:]
        recommended_items.update(top_k_indices)
    
    return len(recommended_items) / total_items


class ModelEvaluator:
    """Comprehensive model evaluation for recommendation systems."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        """Initialize evaluator with k values for ranking metrics."""
        self.k_values = k_values
        
    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate binary classification metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Probabilistic metrics (if scores provided)
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                metrics['auc_pr'] = average_precision_score(y_true, y_scores)
                metrics['log_loss'] = log_loss(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"Could not calculate probabilistic metrics: {e}")
        
        return metrics
    
    def evaluate_ranking(self, 
                        y_true: np.ndarray, 
                        y_scores: np.ndarray) -> Dict[str, float]:
        """Evaluate ranking metrics for recommendation systems."""
        metrics = {}
        
        # Calculate metrics for each k
        for k in self.k_values:
            metrics[f'precision_at_{k}'] = precision_at_k(y_true, y_scores, k)
            metrics[f'recall_at_{k}'] = recall_at_k(y_true, y_scores, k)
            metrics[f'f1_at_{k}'] = f1_at_k(y_true, y_scores, k)
            metrics[f'ndcg_at_{k}'] = ndcg_at_k(y_true, y_scores, k)
            metrics[f'hit_rate_at_{k}'] = hit_rate_at_k(y_true, y_scores, k)
        
        # Overall metrics
        metrics['mean_average_precision'] = mean_average_precision(y_true, y_scores)
        
        return metrics
    
    def evaluate_per_user(self, 
                         user_predictions: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """Evaluate metrics per user."""
        user_metrics = []
        
        for user_id, (y_true, y_scores) in user_predictions.items():
            user_eval = {'user_id': user_id}
            
            # Classification metrics
            y_pred = (y_scores >= 0.5).astype(int)
            user_eval.update(self.evaluate_classification(y_true, y_pred, y_scores))
            
            # Ranking metrics  
            user_eval.update(self.evaluate_ranking(y_true, y_scores))
            
            user_metrics.append(user_eval)
        
        return pd.DataFrame(user_metrics)
    
    def evaluate_global(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate global metrics across all predictions."""
        metrics = {}
        
        # Classification metrics
        metrics.update(self.evaluate_classification(y_true, y_pred, y_scores))
        
        # Ranking metrics (if scores provided)
        if y_scores is not None:
            metrics.update(self.evaluate_ranking(y_true, y_scores))
        
        return metrics
    
    def create_evaluation_report(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray, 
                               y_scores: Optional[np.ndarray] = None,
                               model_name: str = "Model") -> Dict:
        """Create comprehensive evaluation report."""
        report = {
            'model_name': model_name,
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'positive_rate': float(np.mean(y_true))
        }
        
        # Add global metrics
        global_metrics = self.evaluate_global(y_true, y_pred, y_scores)
        report['metrics'] = global_metrics
        
        # Add per-k breakdown for ranking metrics
        if y_scores is not None:
            ranking_summary = {}
            for k in self.k_values:
                ranking_summary[f'k_{k}'] = {
                    'precision': global_metrics.get(f'precision_at_{k}', 0),
                    'recall': global_metrics.get(f'recall_at_{k}', 0),
                    'f1': global_metrics.get(f'f1_at_{k}', 0),
                    'ndcg': global_metrics.get(f'ndcg_at_{k}', 0)
                }
            report['ranking_summary'] = ranking_summary
        
        return report
    
    def compare_models(self, model_reports: List[Dict]) -> pd.DataFrame:
        """Compare multiple models' performance."""
        comparison_data = []
        
        for report in model_reports:
            model_data = {
                'model_name': report['model_name'],
                'total_samples': report['total_samples'],
                'positive_rate': report['positive_rate']
            }
            
            # Add key metrics
            metrics = report.get('metrics', {})
            key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
            for metric in key_metrics:
                model_data[metric] = metrics.get(metric, np.nan)
            
            # Add precision@k metrics
            for k in self.k_values:
                model_data[f'precision_at_{k}'] = metrics.get(f'precision_at_{k}', np.nan)
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by key metrics
        ranking_metrics = ['f1', 'auc_roc', 'precision_at_10']
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df


def calculate_business_metrics(predictions_df: pd.DataFrame, 
                             actual_orders_df: pd.DataFrame,
                             product_values: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Calculate business-relevant metrics for recommendations."""
    metrics = {}
    
    # Revenue impact (if product values provided)
    if product_values is not None:
        predicted_revenue = predictions_df.merge(product_values, on='product_id')['value'].sum()
        actual_revenue = actual_orders_df.merge(product_values, on='product_id')['value'].sum()
        
        metrics['predicted_revenue'] = predicted_revenue
        metrics['actual_revenue'] = actual_revenue
        metrics['revenue_capture_rate'] = predicted_revenue / actual_revenue if actual_revenue > 0 else 0
    
    # Order completion rate
    predicted_orders = predictions_df['order_id'].nunique()
    actual_orders = actual_orders_df['order_id'].nunique()
    metrics['order_coverage'] = predicted_orders / actual_orders if actual_orders > 0 else 0
    
    # Average basket size
    metrics['avg_predicted_basket_size'] = predictions_df.groupby('order_id').size().mean()
    metrics['avg_actual_basket_size'] = actual_orders_df.groupby('order_id').size().mean()
    
    return metrics


# Convenience function for quick evaluation
def quick_evaluate(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_scores: Optional[np.ndarray] = None,
                  k_values: List[int] = [5, 10, 20]) -> None:
    """Quick evaluation with printed results."""
    evaluator = ModelEvaluator(k_values)
    report = evaluator.create_evaluation_report(y_true, y_pred, y_scores)
    
    print(f"\nEvaluation Report for {report['model_name']}")
    print("="*50)
    print(f"Total Samples: {report['total_samples']:,}")
    print(f"Positive Rate: {report['positive_rate']:.3f}")
    print()
    
    print("Classification Metrics:")
    metrics = report['metrics']
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        if metric in metrics:
            print(f"  {metric.upper()}: {metrics[metric]:.3f}")
    
    if 'ranking_summary' in report:
        print("\nRanking Metrics:")
        for k, k_metrics in report['ranking_summary'].items():
            print(f"  {k.upper()}:")
            for metric, value in k_metrics.items():
                print(f"    {metric}: {value:.3f}")
    print("="*50)