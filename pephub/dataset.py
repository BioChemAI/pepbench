"""Load and validate peptide-property datasets stored as CSV files."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

class PepDataset:
    """
    Peptide Dataset class for peptide datasets
    
    Args:
        data_dir (Optional[str]): Directory containing raw data files.
            If None, uses the default raw_data directory in the package.
            Defaults to None (uses package data).
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None
    ):
        """
        Initialize Pep Dataset loader
        
        Args:
            data_dir (Optional[str]): Directory containing raw data files.
                If None, uses the default raw_data directory in the package.
                Defaults to None.
        """
        if data_dir is None:
            # Use package data directory
            import pephub
            package_dir = Path(pephub.__file__).parent
            self.data_dir = package_dir / 'raw_data'
        else:
            self.data_dir = Path(data_dir)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist. "
                f"Please ensure the directory exists and contains CSV files."
            )
    
    def load_dataset(self, dataset_name: str) -> "pd.DataFrame":
        """
        Load a single dataset from CSV file
        
        Args:
            dataset_name (str): Name of dataset file to load.
                Should be filename without extension (e.g., 'AMP', 'SOLU').
        
        Returns:
            pd.DataFrame: Loaded dataset
        
        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If dataset file is invalid or missing required columns
        """
        file_path = self.data_dir / f"{dataset_name}.csv"
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file {file_path} not found. "
                f"Available datasets: {self.list_available_datasets()}"
            )
        
        # Load CSV file
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['id', 'peps', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"Dataset {dataset_name} is missing required columns: {missing_columns}. "
                    f"Expected columns: {required_columns}"
                )
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Dataset file {file_path} is empty")
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available dataset files in the data directory
        
        Returns:
            List[str]: List of available dataset names (without .csv extension)
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        return [f.stem for f in csv_files]
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """
        Get information about a dataset
        
        Args:
            dataset_name (str): Name of dataset
        
        Returns:
            Dict[str, any]: Dataset information including:
                - total_samples: Total number of samples
                - positive_samples: Number of positive samples (label=1)
                - negative_samples: Number of negative samples (label=0)
                - positive_ratio: Ratio of positive samples
                - negative_ratio: Ratio of negative samples
                - avg_sequence_length: Average peptide sequence length
        """
        df = self.load_dataset(dataset_name)
        
        total = len(df)
        positive = (df['label'] == 1.0).sum()
        negative = (df['label'] == 0.0).sum()
        avg_length = df['peps'].str.len().mean()
        
        return {
            'total_samples': total,
            'positive_samples': positive,
            'negative_samples': negative,
            'positive_ratio': positive / total if total > 0 else 0.0,
            'negative_ratio': negative / total if total > 0 else 0.0,
            'avg_sequence_length': avg_length
        }


if __name__ == '__main__':
    """Example usage of PepDataset"""
    from .splitter import split_dataset, split_dataset_by_ratio
    
    print("=" * 80)
    print("Example: List Available Datasets")
    print("=" * 80)
    loader = PepDataset()
    available = loader.list_available_datasets()
    print(f"Available datasets: {available}")
    print()
    
    if available:
        dataset_name = available[0]
        print("=" * 80)
        print(f"Example: Load Dataset '{dataset_name}'")
        print("=" * 80)
        try:
            data = loader.load_dataset(dataset_name)
            print(f"Loaded dataset shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            print()
            
            print("=" * 80)
            print("Example: Get Dataset Info")
            print("=" * 80)
            info = loader.get_dataset_info(dataset_name)
            print("Dataset Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()

