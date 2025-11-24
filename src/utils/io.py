"""
Input/Output utilities for data handling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import yaml
import json
from typing import Any, Dict, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from .logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_dataframe(
    df: pd.DataFrame, 
    filepath: Union[str, Path],
    format: str = "parquet",
    **kwargs
) -> None:
    """Save DataFrame in various formats with optimization."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving DataFrame to {filepath} in {format} format")
    
    if format.lower() == "parquet":
        # Optimize for better compression and performance
        df.to_parquet(
            filepath,
            engine='pyarrow',
            compression='snappy',
            index=False,
            **kwargs
        )
    elif format.lower() == "csv":
        df.to_csv(filepath, index=False, **kwargs)
    elif format.lower() == "pickle":
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Successfully saved {df.shape} to {filepath}")


def load_dataframe(
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Load DataFrame from various formats."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format if not specified
    if format is None:
        format = filepath.suffix.lower().replace('.', '')
    
    logger.info(f"Loading DataFrame from {filepath}")
    
    if format == "parquet":
        df = pd.read_parquet(filepath, engine='pyarrow', **kwargs)
    elif format == "csv":
        df = pd.read_csv(filepath, **kwargs)
    elif format == "pickle" or format == "pkl":
        df = pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Successfully loaded {df.shape} from {filepath}")
    return df


def save_model(model: Any, filepath: Union[str, Path], **kwargs) -> None:
    """Save model using joblib for better compatibility."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {filepath}")
    joblib.dump(model, filepath, **kwargs)
    logger.info(f"Model saved successfully to {filepath}")


def load_model(filepath: Union[str, Path], **kwargs) -> Any:
    """Load model using joblib."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    logger.info(f"Loading model from {filepath}")
    model = joblib.load(filepath, **kwargs)
    logger.info(f"Model loaded successfully from {filepath}")
    return model


def save_pickle(obj: Any, filepath: Union[str, Path], **kwargs) -> None:
    """Save object to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, **kwargs)
    
    logger.info(f"Object saved to {filepath}")


def load_pickle(filepath: Union[str, Path], **kwargs) -> Any:
    """Load object from pickle file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    
    logger.info(f"Object loaded from {filepath}")
    return obj


def save_json(data: Dict, filepath: Union[str, Path], **kwargs) -> None:
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, **kwargs)
    
    logger.info(f"JSON data saved to {filepath}")


def load_json(filepath: Union[str, Path], **kwargs) -> Dict:
    """Load dictionary from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f, **kwargs)
    
    logger.info(f"JSON data loaded from {filepath}")
    return data


def get_file_size(filepath: Union[str, Path]) -> float:
    """Get file size in MB."""
    filepath = Path(filepath)
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        return round(size_mb, 2)
    return 0.0


def ensure_directory(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


class DataFrameOptimizer:
    """Utilities for optimizing DataFrame memory usage."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                if str(col_type).startswith('int'):
                    # Optimize integers
                    min_val = df_optimized[col].min()
                    max_val = df_optimized[col].max()
                    
                    if min_val >= 0:  # Unsigned integers
                        if max_val < 255:
                            df_optimized[col] = df_optimized[col].astype(np.uint8)
                        elif max_val < 65535:
                            df_optimized[col] = df_optimized[col].astype(np.uint16)
                        elif max_val < 4294967295:
                            df_optimized[col] = df_optimized[col].astype(np.uint32)
                    else:  # Signed integers
                        if min_val > -128 and max_val < 127:
                            df_optimized[col] = df_optimized[col].astype(np.int8)
                        elif min_val > -32768 and max_val < 32767:
                            df_optimized[col] = df_optimized[col].astype(np.int16)
                        elif min_val > -2147483648 and max_val < 2147483647:
                            df_optimized[col] = df_optimized[col].astype(np.int32)
                
                elif str(col_type).startswith('float'):
                    # Optimize floats
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert to categorical for low cardinality object columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    @staticmethod
    def memory_usage_report(df: pd.DataFrame) -> Dict:
        """Generate memory usage report for DataFrame."""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        report = {
            'total_memory_mb': total_memory / (1024 * 1024),
            'shape': df.shape,
            'columns': {}
        }
        
        for col in df.columns:
            col_memory = memory_usage[col]
            report['columns'][col] = {
                'memory_mb': col_memory / (1024 * 1024),
                'dtype': str(df[col].dtype),
                'percent_of_total': (col_memory / total_memory) * 100
            }
        
        return report