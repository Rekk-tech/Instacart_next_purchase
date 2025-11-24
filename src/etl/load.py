"""
Load module for saving processed data to various destinations.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import shutil
from ..utils.logging import get_logger, log_execution_time
from ..utils.io import save_dataframe, load_config, ensure_directory

logger = get_logger(__name__)


class DataLoader:
    """Load processed data to various destinations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize loader with configuration."""
        self.config = load_config(config_path)
        self.staging_path = Path(self.config.get('data', {}).get('staging_path', './data/staging'))
        self.processed_path = Path(self.config.get('data', {}).get('processed_path', './data/processed'))
    
    @log_execution_time
    def save_to_staging(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Save cleaned tables to staging area."""
        logger.info("Saving tables to staging area...")
        ensure_directory(self.staging_path)
        
        for table_name, df in tables.items():
            output_path = self.staging_path / f"{table_name}.parquet"
            save_dataframe(df, output_path, format="parquet")
            
            # Log file info
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved {table_name}: {df.shape} -> {file_size_mb:.2f}MB")
    
    @log_execution_time
    def save_to_processed(self, tables: Dict[str, pd.DataFrame], partition_cols: Optional[Dict[str, List[str]]] = None) -> None:
        """Save feature-ready tables to processed area with optional partitioning."""
        logger.info("Saving tables to processed area...")
        ensure_directory(self.processed_path)
        
        if partition_cols is None:
            partition_cols = {}
        
        for table_name, df in tables.items():
            output_path = self.processed_path / f"{table_name}.parquet"
            
            # Check if partitioning is specified for this table
            if table_name in partition_cols:
                partition_columns = partition_cols[table_name]
                logger.info(f"Partitioning {table_name} by {partition_columns}")
                
                # Create partitioned directory structure
                partition_dir = self.processed_path / table_name
                ensure_directory(partition_dir)
                
                # Save with partitioning
                df.to_parquet(
                    partition_dir,
                    engine='pyarrow',
                    partition_cols=partition_columns,
                    compression='snappy',
                    index=False
                )
            else:
                # Save as single file
                save_dataframe(df, output_path, format="parquet")
            
            # Log file info
            total_size = sum(f.stat().st_size for f in output_path.parent.rglob("*.parquet") if table_name in str(f))
            total_size_mb = total_size / (1024 * 1024)
            logger.info(f"Saved {table_name}: {df.shape} -> {total_size_mb:.2f}MB")
    
    @log_execution_time
    def create_data_catalog(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Create a data catalog with metadata about processed tables."""
        logger.info("Creating data catalog...")
        
        catalog = {
            'tables': {},
            'total_memory_mb': 0,
            'total_rows': 0,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        for table_name, df in tables.items():
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            catalog['total_memory_mb'] += memory_mb
            catalog['total_rows'] += len(df)
            
            catalog['tables'][table_name] = {
                'shape': df.shape,
                'memory_mb': round(memory_mb, 2),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': int(df.duplicated().sum())
            }
            
            # Add sample data for small tables
            if len(df) <= 1000:
                catalog['tables'][table_name]['sample_data'] = df.head(5).to_dict('records')
        
        catalog['total_memory_mb'] = round(catalog['total_memory_mb'], 2)
        
        return catalog
    
    @log_execution_time
    def save_data_catalog(self, catalog: Dict) -> None:
        """Save data catalog to processed directory."""
        from ..utils.io import save_json
        
        catalog_path = self.processed_path / "data_catalog.json"
        save_json(catalog, catalog_path)
        logger.info(f"Data catalog saved to {catalog_path}")
    
    @log_execution_time
    def archive_data(self, source_path: Union[str, Path], archive_path: Union[str, Path]) -> None:
        """Archive data to backup location."""
        source_path = Path(source_path)
        archive_path = Path(archive_path)
        
        if not source_path.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return
        
        logger.info(f"Archiving {source_path} to {archive_path}")
        ensure_directory(archive_path.parent)
        
        if source_path.is_file():
            shutil.copy2(source_path, archive_path)
        else:
            shutil.copytree(source_path, archive_path, dirs_exist_ok=True)
        
        logger.info(f"Archive completed: {archive_path}")
    
    def load_pipeline(self, tables: Dict[str, pd.DataFrame], save_staging: bool = True, save_processed: bool = True) -> Dict:
        """Complete load pipeline with catalog creation."""
        logger.info("Starting load pipeline...")
        
        # Save to staging if requested
        if save_staging:
            self.save_to_staging(tables)
        
        # Save to processed if requested  
        if save_processed:
            # Define partitioning strategy for large tables
            partition_cols = {
                'order_products__prior': ['eval_set'],  # If eval_set column exists
                # Add more partitioning as needed
            }
            
            # Filter partition_cols based on actual columns
            filtered_partition_cols = {}
            for table_name, cols in partition_cols.items():
                if table_name in tables:
                    existing_cols = [col for col in cols if col in tables[table_name].columns]
                    if existing_cols:
                        filtered_partition_cols[table_name] = existing_cols
            
            self.save_to_processed(tables, filtered_partition_cols)
        
        # Create and save data catalog
        catalog = self.create_data_catalog(tables)
        self.save_data_catalog(catalog)
        
        logger.info("Load pipeline completed successfully!")
        return catalog


def main():
    """Main load pipeline."""
    try:
        from .extract import DataExtractor
        from .transform import DataTransformer
        
        # Extract and transform data
        extractor = DataExtractor()
        raw_tables = extractor.extract_all_tables()
        
        transformer = DataTransformer()
        cleaned_tables = transformer.transform_all_tables(raw_tables)
        
        # Load data
        loader = DataLoader()
        catalog = loader.load_pipeline(cleaned_tables)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ETL PIPELINE SUMMARY")
        logger.info("="*50)
        logger.info(f"Total tables processed: {len(catalog['tables'])}")
        logger.info(f"Total rows: {catalog['total_rows']:,}")
        logger.info(f"Total memory: {catalog['total_memory_mb']:.2f} MB")
        logger.info("="*50)
        
        for table_name, info in catalog['tables'].items():
            logger.info(f"{table_name}: {info['shape']} ({info['memory_mb']:.2f}MB)")
        
        logger.info("ETL pipeline completed successfully!")
        return catalog
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()