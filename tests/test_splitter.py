"""
Tests for splitter module
"""
import pytest
import pandas as pd
import numpy as np
from pephub.splitter import (
    split_dataset,
    split_dataset_by_ratio,
    split_dataset_by_similarity,
    _check_mmseqs2_available,
)


def create_test_dataframe(n_samples=100, n_positive=50):
    """Create test DataFrame"""
    np.random.seed(42)
    sequences = [f"ACDEFGHIKLMNPQRSTVWY{i}" for i in range(n_samples)]
    labels = [1.0] * n_positive + [0.0] * (n_samples - n_positive)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    sequences = [sequences[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return pd.DataFrame({
        'id': range(n_samples),
        'peps': sequences,
        'label': labels
    })


class TestSplitDataset:
    """Tests for split_dataset function"""
    
    def test_basic_split_no_val(self):
        """Test basic split without validation set"""
        data = create_test_dataframe(100, 50)
        train, test, val = split_dataset(data, test_size=0.2, random_state=42)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert val is None
        assert len(train) + len(test) == len(data)
        assert len(train) > 0
        assert len(test) > 0
    
    def test_split_with_val(self):
        """Test split with validation set"""
        data = create_test_dataframe(100, 50)
        train, test, val = split_dataset(
            data, test_size=0.2, val_size=0.1, random_state=42
        )
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert len(train) + len(test) + len(val) == len(data)
    
    def test_stratified_split(self):
        """Test stratified split maintains class proportions"""
        data = create_test_dataframe(100, 50)
        original_ratio = (data['label'] == 1.0).sum() / len(data)
        
        train, test, val = split_dataset(
            data, test_size=0.2, val_size=0.1, random_state=42, stratify=True
        )
        
        train_ratio = (train['label'] == 1.0).sum() / len(train)
        test_ratio = (test['label'] == 1.0).sum() / len(test)
        val_ratio = (val['label'] == 1.0).sum() / len(val)
        
        # Ratios should be approximately equal
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1
        assert abs(val_ratio - original_ratio) < 0.1
    
    def test_invalid_input(self):
        """Test invalid input handling"""
        data = create_test_dataframe(10, 5)
        
        with pytest.raises(TypeError):
            split_dataset("not a dataframe")
        
        with pytest.raises(ValueError, match="label"):
            split_dataset(pd.DataFrame({'peps': ['A'] * 10}))
        
        with pytest.raises(ValueError, match="test_size"):
            split_dataset(data, test_size=1.5)
        
        with pytest.raises(ValueError, match="val_size"):
            split_dataset(data, test_size=0.2, val_size=0.9)


class TestSplitDatasetByRatio:
    """Tests for split_dataset_by_ratio function"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        data = create_test_dataframe(100, 50)
        train, test, val = split_dataset_by_ratio(
            data, test_size=0.2, val_size=0.1, random_state=42
        )
        
        assert len(train) + len(test) + len(val) == len(data)


@pytest.mark.skipif(
    not _check_mmseqs2_available(),
    reason="MMseqs2 is not available"
)
class TestSplitDatasetBySimilarity:
    """Tests for split_dataset_by_similarity function (requires MMseqs2)"""
    
    def test_basic_split(self):
        """Test basic similarity-based split"""
        data = create_test_dataframe(50, 25)
        train, test, val = split_dataset_by_similarity(
            data, test_size=0.2, random_state=42
        )
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert val is None
        assert len(train) + len(test) == len(data)
    
    def test_split_with_val(self):
        """Test similarity-based split with validation set"""
        data = create_test_dataframe(100, 50)
        train, test, val = split_dataset_by_similarity(
            data, test_size=0.2, val_size=0.1, random_state=42
        )
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        total = len(train) + len(test) + len(val)
        assert total == len(data)
    
    def test_invalid_input(self):
        """Test invalid input handling"""
        data = create_test_dataframe(10, 5)
        
        with pytest.raises(ValueError, match="peps"):
            split_dataset_by_similarity(
                pd.DataFrame({'label': [1.0] * 10})
            )

