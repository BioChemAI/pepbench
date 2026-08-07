"""
Tests for featurizer module
"""
import pytest
import numpy as np
from pephub.featurizer import PeptideFeaturizer


class TestPeptideFeaturizer:
    """Tests for PeptideFeaturizer class"""
    
    def test_descriptor_extraction(self):
        """Test descriptor feature extraction"""
        featurizer = PeptideFeaturizer(feature_type='descriptor')
        features = featurizer.transform("ACDEFGHIKLMNPQRSTVWY")
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 1
        assert features.dtype in [np.float32, np.float64]
    
    def test_descriptor_batch(self):
        """Test batch descriptor extraction"""
        featurizer = PeptideFeaturizer(feature_type='descriptor')
        peptides = ["ACDEFG", "GHIKLM", "NPQRST"]
        features = featurizer.transform(peptides)
        
        assert isinstance(features, list)
        assert len(features) == 3
        assert all(isinstance(f, np.ndarray) for f in features)
    
    def test_specific_descriptors(self):
        """Test extraction of specific descriptors"""
        featurizer = PeptideFeaturizer(
            feature_type='descriptor',
            descriptor_list=['molecular_weight', 'isoelectric_point', 'charge']
        )
        features = featurizer.transform("ACDEFGHIKLMNPQRSTVWY")
        
        assert isinstance(features, np.ndarray)
        assert len(features) >= 3
    
    def test_onehot_encoding(self):
        """Test one-hot encoding"""
        featurizer = PeptideFeaturizer(feature_type='onehot')
        features = featurizer.transform("ACDEFG")
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 2
        assert features.shape[0] == len("ACDEFG")
    
    def test_onehot_with_padding(self):
        """Test one-hot encoding with padding"""
        featurizer = PeptideFeaturizer(
            feature_type='onehot',
            padding_len=20
        )
        features = featurizer.transform("ACDEFG")
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 20
    
    def test_integer_encoding(self):
        """Test integer encoding"""
        featurizer = PeptideFeaturizer(feature_type='integer')
        features = featurizer.transform("ACDEFG")
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 1
        assert features.dtype in [np.int32, np.int64]
    
    def test_blosum62_encoding(self):
        """Test BLOSUM62 encoding"""
        featurizer = PeptideFeaturizer(feature_type='blosum62')
        features = featurizer.transform("ACDEFG")
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 2
        assert features.shape[1] == 21  # 20 amino acids + 1 for modifications
    
    def test_invalid_feature_type(self):
        """Test invalid feature type"""
        with pytest.raises(ValueError, match="feature_type"):
            PeptideFeaturizer(feature_type='invalid')
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        featurizer = PeptideFeaturizer(feature_type='onehot')
        peptides = ["ACDEFG", "GHIKLM"]
        features = featurizer.fit_transform(peptides)
        
        assert isinstance(features, list)
        assert len(features) == 2
    
    @pytest.mark.skipif(
        not PeptideFeaturizer.__module__.startswith('pephub'),
        reason="ESM dependencies not available"
    )
    def test_esm_encoding(self):
        """Test ESM encoding (if available)"""
        try:
            featurizer = PeptideFeaturizer(
                feature_type='esm',
                esm_model_name='facebook/esm2_t6_8M_UR50D',
                esm_pooling='mean'
            )
            features = featurizer.transform("ACDEFG")
            
            assert isinstance(features, np.ndarray)
            assert len(features.shape) == 1
        except ImportError:
            pytest.skip("ESM dependencies not available")

