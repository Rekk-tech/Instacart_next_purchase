"""
Extract module for Instacart dataset.
Handles data ingestion from various sources with schema validation.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from ..utils.logging import get_logger
from ..utils.io import save_dataframe, load_config

logger = get_logger(__name__)


class DataExtractor:
    """Extract data from various sources with validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize extractor with configuration."""
        self.config = load_config(config_path)
        self.raw_data_path = Path(self.config.get('data', {}).get('raw_path', './data/raw'))
        self.expected_schemas = self._get_expected_schemas()
        
    def _get_expected_schemas(self) -> Dict:
        """Define expected schemas for validation."""
        return {
            'orders': {
                'order_id': 'int64',
                'user_id': 'int64', 
                'eval_set': 'object',
                'order_number': 'int64',
                'order_dow': 'int64',
                'order_hour_of_day': 'int64',
                'days_since_prior_order': 'float64'
            },
            'order_products__prior': {
                'order_id': 'int64',
                'product_id': 'int64',
                'add_to_cart_order': 'int64',
                'reordered': 'int64'
            },
            'order_products__train': {
                'order_id': 'int64', 
                'product_id': 'int64',
                'add_to_cart_order': 'int64',
                'reordered': 'int64'
            },
            'products': {
                'product_id': 'int64',
                'product_name': 'object',
                'aisle_id': 'int64',
                'department_id': 'int64'
            },
            'aisles': {
                'aisle_id': 'int64',
                'aisle': 'object'
            },
            'departments': {
                'department_id': 'int64',
                'department': 'object'
            }
        }
    
    def validate_schema(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate dataframe schema against expected schema."""
        expected = self.expected_schemas.get(table_name, {})
        
        # Check if all expected columns exist
        missing_cols = set(expected.keys()) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns in {table_name}: {missing_cols}")
            return False
            
        # Check data types
        for col, expected_dtype in expected.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    logger.warning(f"Column {col} in {table_name}: expected {expected_dtype}, got {actual_dtype}")
        
        logger.info(f"Schema validation passed for {table_name}")
        return True
    
    def extract_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Extract data from CSV file with validation."""
        filepath = self.raw_data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading {filename}...")
        
        try:
            # Load CSV with optimized settings
            df = pd.read_csv(
                filepath,
                dtype_backend='pyarrow',  # Use pyarrow for better performance
                **kwargs
            )
            
            # Validate schema
            table_name = filename.replace('.csv', '')
            if not self.validate_schema(df, table_name):
                logger.warning(f"Schema validation failed for {table_name}")
            
            logger.info(f"Successfully loaded {filename}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def extract_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Extract all Instacart tables."""
        tables = {}
        
        # Core tables
        table_files = [
            'orders.csv',
            'order_products__prior.csv', 
            'order_products__train.csv',
            'products.csv',
            'aisles.csv',
            'departments.csv'
        ]
        
        for filename in table_files:
            table_name = filename.replace('.csv', '')
            tables[table_name] = self.extract_csv(filename)
            
        return tables
    
    def get_data_summary(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for extracted data."""
        summary = {}
        
        for name, df in tables.items():
            summary[name] = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'columns': list(df.columns)
            }
            
        return summary


def main():
    """Main extraction pipeline."""
    try:
        # Initialize extractor
        extractor = DataExtractor()
        
        # Extract all tables
        logger.info("Starting data extraction...")
        tables = extractor.extract_all_tables()
        
        # Generate summary
        summary = extractor.get_data_summary(tables)
        
        # Log summary
        for table_name, stats in summary.items():
            logger.info(f"{table_name}: {stats['shape']}, "
                       f"{stats['memory_usage_mb']:.2f}MB, "
                       f"{stats['missing_values']} missing values")
        
        logger.info("Data extraction completed successfully!")
        return tables
        
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()