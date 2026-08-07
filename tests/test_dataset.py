"""
Tests for dataset module
"""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from pephub.dataset import PepDataset


def create_test_csv(file_path, n_samples=10):
    """Create a test CSV file"""
    data = {
        'id': range(n_samples),
        'peps': [f"ACDEFGHIKLMNPQRSTVWY{i}" for i in range(n_samples)],
        'label': [1.0] * (n_samples // 2) + [0.0] * (n_samples - n_samples // 2)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


class TestPepDataset:
    """Tests for PepDataset class"""
    
    def test_list_available_datasets(self):
        """Test listing available datasets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            create_test_csv(os.path.join(tmpdir, "test1.csv"), 10)
            create_test_csv(os.path.join(tmpdir, "test2.csv"), 10)
            
            loader = PepDataset(data_dir=tmpdir)
            datasets = loader.list_available_datasets()
            
            assert len(datasets) == 2
            assert "test1" in datasets
            assert "test2" in datasets
    
    def test_load_dataset(self):
        """Test loading a dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_csv(os.path.join(tmpdir, "test.csv"), 10)
            
            loader = PepDataset(data_dir=tmpdir)
            data = loader.load_dataset("test")
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 10
            assert 'id' in data.columns
            assert 'peps' in data.columns
            assert 'label' in data.columns
    
    def test_load_nonexistent_dataset(self):
        """Test loading non-existent dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PepDataset(data_dir=tmpdir)
            
            with pytest.raises(FileNotFoundError):
                loader.load_dataset("nonexistent")
    
    def test_get_dataset_info(self):
        """Test getting dataset information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_csv(os.path.join(tmpdir, "test.csv"), 10)
            
            loader = PepDataset(data_dir=tmpdir)
            info = loader.get_dataset_info("test")
            
            assert 'total_samples' in info
            assert 'positive_samples' in info
            assert 'negative_samples' in info
            assert 'positive_ratio' in info
            assert 'negative_ratio' in info
            assert 'avg_sequence_length' in info
            assert info['total_samples'] == 10
    
    def test_missing_columns(self):
        """Test dataset with missing required columns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV without required columns
            df = pd.DataFrame({'wrong_col': [1, 2, 3]})
            df.to_csv(os.path.join(tmpdir, "test.csv"), index=False)
            
            loader = PepDataset(data_dir=tmpdir)
            
            with pytest.raises(ValueError, match="missing required columns"):
                loader.load_dataset("test")
    
    def test_nonexistent_data_dir(self):
        """Test with non-existent data directory"""
        with pytest.raises(FileNotFoundError):
            PepDataset(data_dir="nonexistent_directory")

