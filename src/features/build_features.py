"""
Feature engineering orchestrator that combines all feature types.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from ..utils.logging import get_logger, log_execution_time
from ..utils.io import load_config, load_dataframe, save_dataframe, ensure_directory
from .fe_user_item import UserItemFeatureEngine
from .fe_time import TimeFeatureEngine

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """Main feature engineering orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature builder with configuration."""
        self.config = load_config(config_path)
        self.staging_path = Path(self.config.get('data', {}).get('staging_path', './data/staging'))
        self.processed_path = Path(self.config.get('data', {}).get('processed_path', './data/processed'))
        
        # Initialize feature engines
        self.user_item_engine = UserItemFeatureEngine(config_path)
        self.time_engine = TimeFeatureEngine(config_path)
        
        # Feature configuration
        self.feature_config = self.config.get('features', {})
    
    @log_execution_time
    def load_staging_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from staging area."""
        logger.info("Loading data from staging area...")
        
        tables = {}
        required_tables = [
            'orders', 'order_products__prior', 'order_products__train',
            'enriched_products', 'aisles', 'departments'
        ]
        
        for table_name in required_tables:
            file_path = self.staging_path / f"{table_name}.parquet"
            if file_path.exists():
                tables[table_name] = load_dataframe(file_path)
                logger.info(f"Loaded {table_name}: {tables[table_name].shape}")
            else:
                logger.warning(f"Table not found: {table_name}")
        
        return tables
    
    @log_execution_time
    def create_train_test_split(self, orders: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split based on eval_set."""
        logger.info("Creating train/test split...")
        
        # Split based on eval_set column
        train_orders = orders[orders['eval_set'] == 'prior'].copy()
        test_orders = orders[orders['eval_set'].isin(['train', 'test'])].copy()
        
        logger.info(f"Train orders: {train_orders.shape}")
        logger.info(f"Test orders: {test_orders.shape}")
        
        return train_orders, test_orders
    
    @log_execution_time
    def build_user_features(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build user-level aggregated features."""
        logger.info("Building user features...")
        
        orders = tables['orders']
        order_products = tables['order_products__prior']
        products = tables['enriched_products']
        
        # Get prior orders only for feature calculation
        prior_orders = orders[orders['eval_set'] == 'prior'].copy()
        
        # User aggregated features
        user_features = []
        
        # Basic user statistics
        user_stats = prior_orders.groupby('user_id').agg({
            'order_number': ['count', 'max'],
            'days_since_prior_order': ['mean', 'std', 'max'],
            'order_dow': lambda x: x.mode().iloc[0] if len(x) > 0 else x.iloc[0],
            'order_hour_of_day': lambda x: x.mode().iloc[0] if len(x) > 0 else x.iloc[0]
        }).round(2)
        
        user_stats.columns = [
            'total_orders', 'max_order_number', 'avg_days_between_orders', 
            'std_days_between_orders', 'max_days_between_orders',
            'favorite_dow', 'favorite_hour'
        ]
        
        user_features.append(user_stats)
        
        # User-product interaction features
        user_product_stats = (
            prior_orders.merge(order_products, on='order_id')
            .groupby('user_id')
            .agg({
                'product_id': 'nunique',
                'add_to_cart_order': ['mean', 'std'],
                'reordered': ['mean', 'sum']
            }).round(2)
        )
        
        user_product_stats.columns = [
            'unique_products', 'avg_basket_position', 'std_basket_position',
            'reorder_ratio', 'total_reorders'
        ]
        
        user_features.append(user_product_stats)
        
        # Department/aisle diversity
        user_diversity = (
            prior_orders
            .merge(order_products, on='order_id')
            .merge(products[['product_id', 'department_id', 'aisle_id']], on='product_id')
            .groupby('user_id')
            .agg({
                'department_id': 'nunique',
                'aisle_id': 'nunique'
            })
        )
        
        user_diversity.columns = ['unique_departments', 'unique_aisles']
        user_features.append(user_diversity)
        
        # Combine all user features
        user_features_df = user_features[0]
        for df in user_features[1:]:
            user_features_df = user_features_df.join(df, how='left')
        
        # Fill missing values
        user_features_df = user_features_df.fillna(0)
        
        # Add derived features
        user_features_df['avg_basket_size'] = (
            user_features_df['total_reorders'] + 
            (user_features_df['unique_products'] - user_features_df['total_reorders'])
        ) / user_features_df['total_orders']
        
        user_features_df['customer_loyalty'] = (
            user_features_df['reorder_ratio'] * user_features_df['total_orders']
        ).fillna(0)
        
        logger.info(f"User features created: {user_features_df.shape}")
        return user_features_df.reset_index()
    
    @log_execution_time 
    def build_product_features(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build product-level aggregated features."""
        logger.info("Building product features...")
        
        orders = tables['orders']
        order_products = tables['order_products__prior']
        products = tables['enriched_products']
        
        # Get prior orders only
        prior_orders = orders[orders['eval_set'] == 'prior'].copy()
        
        # Product aggregated features
        product_stats = (
            prior_orders
            .merge(order_products, on='order_id')
            .groupby('product_id')
            .agg({
                'user_id': 'nunique',
                'order_id': 'count',
                'add_to_cart_order': ['mean', 'std'],
                'reordered': ['mean', 'sum', 'count'],
                'order_dow': lambda x: x.mode().iloc[0] if len(x) > 0 else x.iloc[0],
                'order_hour_of_day': lambda x: x.mode().iloc[0] if len(x) > 0 else x.iloc[0]
            }).round(2)
        )
        
        product_stats.columns = [
            'unique_users', 'total_orders', 'avg_cart_position', 'std_cart_position',
            'reorder_probability', 'total_reorders', 'total_purchases',
            'popular_dow', 'popular_hour'
        ]
        
        # Add product metadata
        product_features = products[['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department']].merge(
            product_stats.reset_index(), on='product_id', how='left'
        ).fillna(0)
        
        # Calculate popularity scores
        product_features['popularity_score'] = (
            product_features['unique_users'] * product_features['reorder_probability']
        ).fillna(0)
        
        # Department/aisle level popularity
        dept_popularity = product_features.groupby('department_id')['unique_users'].mean()
        aisle_popularity = product_features.groupby('aisle_id')['unique_users'].mean()
        
        product_features['department_popularity'] = product_features['department_id'].map(dept_popularity)
        product_features['aisle_popularity'] = product_features['aisle_id'].map(aisle_popularity)
        
        logger.info(f"Product features created: {product_features.shape}")
        return product_features
    
    @log_execution_time
    def build_user_item_matrix(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build user-item interaction matrix with features."""
        logger.info("Building user-item interaction matrix...")
        
        # Use the specialized user-item feature engine
        user_item_features = self.user_item_engine.create_user_item_features(tables)
        
        logger.info(f"User-item features created: {user_item_features.shape}")
        return user_item_features
    
    @log_execution_time
    def build_time_features(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Build time-based features."""
        logger.info("Building time-based features...")
        
        # Use the specialized time feature engine
        time_features = self.time_engine.create_time_features(tables)
        
        logger.info("Time features created")
        return time_features
    
    @log_execution_time
    def create_final_dataset(self, 
                           user_features: pd.DataFrame,
                           product_features: pd.DataFrame, 
                           user_item_features: pd.DataFrame,
                           tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create final dataset for modeling."""
        logger.info("Creating final dataset...")
        
        # Start with user-item interactions
        final_dataset = user_item_features.copy()
        
        # Add user features
        final_dataset = final_dataset.merge(
            user_features, on='user_id', how='left', suffixes=('', '_user')
        )
        
        # Add product features
        final_dataset = final_dataset.merge(
            product_features[['product_id', 'popularity_score', 'reorder_probability', 
                            'avg_cart_position', 'department_popularity', 'aisle_popularity']], 
            on='product_id', how='left', suffixes=('', '_product')
        )
        
        # Add basic order information
        orders = tables['orders']
        final_dataset = final_dataset.merge(
            orders[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']], 
            on='order_id', how='left'
        )
        
        # Fill missing values
        numeric_columns = final_dataset.select_dtypes(include=[np.number]).columns
        final_dataset[numeric_columns] = final_dataset[numeric_columns].fillna(0)
        
        # Feature selection based on config
        if 'feature_selection' in self.config.get('preprocessing', {}):
            final_dataset = self._select_features(final_dataset)
        
        logger.info(f"Final dataset created: {final_dataset.shape}")
        return final_dataset
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature selection based on configuration."""
        selection_config = self.config['preprocessing']['feature_selection']
        method = selection_config.get('method', 'all')
        k_best = selection_config.get('k_best', len(df.columns))
        
        if method == 'all' or k_best >= len(df.columns):
            return df
        
        logger.info(f"Selecting top {k_best} features using {method}")
        
        # Basic feature selection (could be enhanced with more sophisticated methods)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > k_best:
            # Simple variance-based selection
            variances = df[numeric_cols].var()
            top_features = variances.nlargest(k_best).index.tolist()
            
            # Keep non-numeric columns and selected numeric columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            selected_cols = non_numeric_cols + top_features
            
            return df[selected_cols]
        
        return df
    
    def build_all_features(self) -> Dict[str, pd.DataFrame]:
        """Build all features and create final datasets."""
        logger.info("Starting complete feature engineering pipeline...")
        
        # Load staging data
        tables = self.load_staging_data()
        
        # Build different feature types
        user_features = self.build_user_features(tables)
        product_features = self.build_product_features(tables)
        user_item_features = self.build_user_item_matrix(tables)
        
        # Create final dataset
        final_dataset = self.create_final_dataset(
            user_features, product_features, user_item_features, tables
        )
        
        # Prepare output dictionary
        feature_tables = {
            'user_features': user_features,
            'product_features': product_features,  
            'user_item_features': user_item_features,
            'final_dataset': final_dataset
        }
        
        # Save features to processed directory
        ensure_directory(self.processed_path)
        
        for table_name, df in feature_tables.items():
            output_path = self.processed_path / f"{table_name}.parquet"
            save_dataframe(df, output_path)
            logger.info(f"Saved {table_name}: {df.shape}")
        
        logger.info("Feature engineering pipeline completed!")
        return feature_tables


def main():
    """Main feature engineering pipeline."""
    try:
        builder = FeatureBuilder()
        feature_tables = builder.build_all_features()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*50)
        
        for table_name, df in feature_tables.items():
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"{table_name}: {df.shape} ({memory_mb:.2f}MB)")
            
        logger.info("="*50)
        logger.info("Feature engineering completed successfully!")
        
        return feature_tables
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()