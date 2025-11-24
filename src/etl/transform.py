"""
Transform module for data cleaning and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
from ..utils.logging import get_logger, log_execution_time
from ..utils.io import load_config, DataFrameOptimizer

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class DataTransformer:
    """Transform and clean raw data for staging."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize transformer with configuration."""
        self.config = load_config(config_path)
        self.optimizer = DataFrameOptimizer()
        
    @log_execution_time
    def clean_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance orders data."""
        logger.info("Cleaning orders data...")
        df = orders.copy()
        
        # Convert to datetime if needed  
        df['order_dow'] = df['order_dow'].astype('int8')
        df['order_hour_of_day'] = df['order_hour_of_day'].astype('int8')
        
        # Handle missing days_since_prior_order (first orders)
        df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0)
        
        # Create additional time features
        df['is_weekend'] = (df['order_dow'].isin([0, 6])).astype('int8')
        
        # Create time buckets
        df['hour_bucket'] = pd.cut(
            df['order_hour_of_day'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        ).astype('category')
        
        # Days since prior order buckets
        df['days_bucket'] = pd.cut(
            df['days_since_prior_order'],
            bins=[-1, 0, 7, 14, 30, np.inf],
            labels=['first_order', 'week', 'two_weeks', 'month', 'long_time'],
            include_lowest=True
        ).astype('category')
        
        # Optimize dtypes
        df = self.optimizer.optimize_dtypes(df)
        
        logger.info(f"Orders cleaned: {df.shape}")
        return df
    
    @log_execution_time
    def clean_order_products(self, order_products: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Clean order products data (prior or train)."""
        logger.info(f"Cleaning {table_name} data...")
        df = order_products.copy()
        
        # Convert boolean reordered to int8
        df['reordered'] = df['reordered'].astype('int8')
        df['add_to_cart_order'] = df['add_to_cart_order'].astype('int16')
        
        # Sort by order_id and add_to_cart_order for consistency
        df = df.sort_values(['order_id', 'add_to_cart_order'])
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['order_id', 'product_id']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate product-order pairs in {table_name}")
            df = df.drop_duplicates(subset=['order_id', 'product_id'], keep='first')
        
        # Optimize dtypes
        df = self.optimizer.optimize_dtypes(df)
        
        logger.info(f"{table_name} cleaned: {df.shape}")
        return df
    
    @log_execution_time 
    def clean_products(self, products: pd.DataFrame) -> pd.DataFrame:
        """Clean products data."""
        logger.info("Cleaning products data...")
        df = products.copy()
        
        # Clean product names
        df['product_name'] = df['product_name'].str.strip()
        
        # Handle missing product names
        missing_names = df['product_name'].isnull().sum()
        if missing_names > 0:
            logger.warning(f"Found {missing_names} products with missing names")
            df['product_name'] = df['product_name'].fillna('Unknown Product')
        
        # Optimize dtypes  
        df = self.optimizer.optimize_dtypes(df)
        
        logger.info(f"Products cleaned: {df.shape}")
        return df
    
    @log_execution_time
    def clean_aisles(self, aisles: pd.DataFrame) -> pd.DataFrame:
        """Clean aisles data."""
        logger.info("Cleaning aisles data...")
        df = aisles.copy()
        
        # Clean aisle names
        df['aisle'] = df['aisle'].str.strip().str.title()
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values in aisles")
        
        # Optimize dtypes
        df = self.optimizer.optimize_dtypes(df)
        
        logger.info(f"Aisles cleaned: {df.shape}")
        return df
    
    @log_execution_time
    def clean_departments(self, departments: pd.DataFrame) -> pd.DataFrame:
        """Clean departments data."""
        logger.info("Cleaning departments data...")
        df = departments.copy()
        
        # Clean department names
        df['department'] = df['department'].str.strip().str.title()
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values in departments")
        
        # Optimize dtypes
        df = self.optimizer.optimize_dtypes(df)
        
        logger.info(f"Departments cleaned: {df.shape}")
        return df
    
    @log_execution_time
    def create_enriched_products(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create enriched products table with aisle and department info."""
        logger.info("Creating enriched products table...")
        
        # Join products with aisles and departments
        enriched_products = (
            tables['products']
            .merge(tables['aisles'], on='aisle_id', how='left')
            .merge(tables['departments'], on='department_id', how='left')
        )
        
        # Add product hierarchy level
        enriched_products['product_hierarchy'] = (
            enriched_products['department'] + ' > ' + enriched_products['aisle']
        )
        
        logger.info(f"Enriched products created: {enriched_products.shape}")
        return enriched_products
    
    @log_execution_time
    def validate_data_integrity(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate data integrity and return issues."""
        logger.info("Validating data integrity...")
        issues = {}
        
        # Check for orphaned records
        order_ids_in_products_prior = set(tables['order_products__prior']['order_id'])
        order_ids_in_orders = set(tables['orders']['order_id'])
        orphaned_orders_prior = order_ids_in_products_prior - order_ids_in_orders
        
        if orphaned_orders_prior:
            issues['order_products__prior'] = [
                f"Found {len(orphaned_orders_prior)} orphaned order_ids not in orders table"
            ]
        
        # Check product references
        product_ids_in_orders = set(tables['order_products__prior']['product_id']).union(
            set(tables['order_products__train']['product_id'])
        )
        product_ids_in_products = set(tables['products']['product_id'])
        orphaned_products = product_ids_in_orders - product_ids_in_products
        
        if orphaned_products:
            issues['products'] = [
                f"Found {len(orphaned_products)} product_ids referenced but not in products table"
            ]
        
        # Check aisle/department references
        aisle_ids_in_products = set(tables['products']['aisle_id'])
        aisle_ids_in_aisles = set(tables['aisles']['aisle_id'])
        missing_aisles = aisle_ids_in_products - aisle_ids_in_aisles
        
        if missing_aisles:
            issues['aisles'] = [
                f"Found {len(missing_aisles)} aisle_ids referenced but not in aisles table"
            ]
        
        dept_ids_in_products = set(tables['products']['department_id'])
        dept_ids_in_depts = set(tables['departments']['department_id'])
        missing_depts = dept_ids_in_products - dept_ids_in_depts
        
        if missing_depts:
            issues['departments'] = [
                f"Found {len(missing_depts)} department_ids referenced but not in departments table"
            ]
        
        if not issues:
            logger.info("Data integrity validation passed!")
        else:
            for table, table_issues in issues.items():
                for issue in table_issues:
                    logger.warning(f"{table}: {issue}")
        
        return issues
    
    def transform_all_tables(self, raw_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform all tables and return cleaned versions."""
        logger.info("Starting data transformation...")
        
        cleaned_tables = {}
        
        # Clean individual tables
        cleaned_tables['orders'] = self.clean_orders(raw_tables['orders'])
        cleaned_tables['order_products__prior'] = self.clean_order_products(
            raw_tables['order_products__prior'], 'order_products__prior'
        )
        cleaned_tables['order_products__train'] = self.clean_order_products(
            raw_tables['order_products__train'], 'order_products__train'
        )
        cleaned_tables['products'] = self.clean_products(raw_tables['products'])
        cleaned_tables['aisles'] = self.clean_aisles(raw_tables['aisles'])
        cleaned_tables['departments'] = self.clean_departments(raw_tables['departments'])
        
        # Create enriched tables
        cleaned_tables['enriched_products'] = self.create_enriched_products(cleaned_tables)
        
        # Validate data integrity
        self.validate_data_integrity(cleaned_tables)
        
        logger.info("Data transformation completed!")
        return cleaned_tables


def main():
    """Main transformation pipeline."""
    try:
        from .extract import DataExtractor
        from ..utils.io import save_dataframe
        
        # Extract data
        extractor = DataExtractor()
        raw_tables = extractor.extract_all_tables()
        
        # Transform data
        transformer = DataTransformer()
        cleaned_tables = transformer.transform_all_tables(raw_tables)
        
        # Save to staging
        staging_path = Path("./data/staging")
        staging_path.mkdir(parents=True, exist_ok=True)
        
        for table_name, df in cleaned_tables.items():
            output_path = staging_path / f"{table_name}.parquet"
            save_dataframe(df, output_path, format="parquet")
            logger.info(f"Saved {table_name} to staging: {df.shape}")
        
        logger.info("Transformation pipeline completed successfully!")
        return cleaned_tables
        
    except Exception as e:
        logger.error(f"Transformation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()