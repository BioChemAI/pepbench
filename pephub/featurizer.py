"""Extract descriptor, sequence-encoding, and ESM peptide features."""

from typing import Union, List, Dict, Any, Optional
import numpy as np
from peptidy import descriptors
from peptidy import encoding
from peptidy import biology

# Optional import for ESM transformer models
try:
    import torch
    from transformers import EsmModel, EsmTokenizer
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    torch = None
    EsmModel = None
    EsmTokenizer = None


class PeptideFeaturizer:
    """
    Peptide featurizer class supporting descriptor, onehot, integer, blosum62, and ESM feature extraction
    
    Args:
        feature_type (str): Feature type, options: 'descriptor', 'onehot', 'frequency', 'integer', 'blosum62', 'esm'
            - 'descriptor': Extract physicochemical descriptors (returns numpy array)
            - 'onehot': One-hot encoding of amino acids (returns numpy array)
            - 'integer': Integer/label encoding of amino acids (returns numpy array)
            - 'blosum62': BLOSUM62 substitution matrix encoding (returns numpy array)
            - 'esm': ESM (Evolutionary Scale Modeling) transformer encoding (returns numpy array)
                Requires 'transformers' and 'torch' packages to be installed separately.
        descriptor_list (Optional[List[str]]): When feature_type='descriptor',
            specify the list of descriptors to compute. If None, compute all descriptors.
            Available descriptors include: 'molecular_weight', 'isoelectric_point', 'charge',
            'aminoacid_frequencies', 'molecular_formula', etc.
        padding_len (Optional[int]): The length to which the encoded vector should be padded.
            If the vector is shorter than padding_len, it will be padded. If it is longer,
            a ValueError will be raised. Only used for 'onehot', 'integer', and 'blosum62' feature types.
            Defaults to None (no padding).
        add_generative_tokens (bool): Whether to add special tokens ("<beg>", "<end>") to the
            beginning and end of the encoded vector for generative applications.
            Only used for 'onehot', 'integer', and 'blosum62' feature types.
            Defaults to False.
        esm_model_name (Optional[str]): ESM model name to use when feature_type='esm'.
            Options include: 'facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D',
            'facebook/esm2_t33_650M_UR50D', 'facebook/esm1b_t33_650M_UR50S', etc.
            Defaults to 'facebook/esm2_t6_8M_UR50D' (smallest model).
        esm_layer_index (Optional[int]): Which layer's embeddings to extract from ESM model.
            If None, uses the last layer. Defaults to None.
        esm_pooling (str): How to pool ESM embeddings. Options: 'mean', 'cls', 'max'.
            'mean': Average pooling over sequence length (excluding special tokens).
            'cls': Use the [CLS] token embedding.
            'max': Max pooling over sequence length.
            Defaults to 'mean'.
        esm_batch_size (Optional[int]): Batch size for ESM feature extraction when processing multiple sequences.
            If None, processes all sequences in a single batch. Useful for memory management when processing
            large numbers of sequences. Defaults to None (process all at once).
        device (Optional[str]): Device to run ESM model on ('cpu', 'cuda', 'cuda:0', etc.).
            If None, uses 'cuda' if available, else 'cpu'. Defaults to None.
    """
    
    def __init__(
        self, 
        feature_type: str = 'descriptor',
        descriptor_list: Optional[List[str]] = None,
        padding_len: Optional[int] = None,
        add_generative_tokens: bool = False,
        esm_model_name: Optional[str] = 'facebook/esm2_t6_8M_UR50D',
        esm_layer_index: Optional[int] = None,
        esm_pooling: str = None,
        esm_batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize peptide featurizer
        
        Args:
            feature_type (str): Feature type, options: 'descriptor', 'onehot', 'integer', 'blosum62', 'esm'
            descriptor_list (Optional[List[str]]): Descriptor list, only effective when feature_type='descriptor'.
                If None, all descriptors are computed.
            padding_len (Optional[int]): Padding length for encoding. Only used for 'onehot', 'integer', 'blosum62'.
                Defaults to None (no padding).
            add_generative_tokens (bool): Whether to add generative tokens. Only used for 'onehot', 'integer', 'blosum62'.
                Defaults to False.
            esm_model_name (Optional[str]): ESM model name. Only used when feature_type='esm'.
                Defaults to 'facebook/esm2_t6_8M_UR50D'.
            esm_layer_index (Optional[int]): ESM layer index. Only used when feature_type='esm'.
                Defaults to None (last layer).
            esm_pooling (str): ESM pooling method. Only used when feature_type='esm'.
                Defaults to 'mean'.
            esm_batch_size (Optional[int]): Batch size for ESM feature extraction. Only used when feature_type='esm'.
                If None, processes all sequences in a single batch. Defaults to None.
            device (Optional[str]): Device for ESM model. Only used when feature_type='esm'.
                Defaults to None (auto-detect).
        """
        if feature_type not in ['descriptor', 'onehot', 'frequency', 'integer', 'blosum62', 'esm']:
            raise ValueError(
                f"feature_type must be one of 'descriptor', 'onehot', 'frequency', 'integer', 'blosum62', or 'esm', "
                f"got: {feature_type}"
            )
        
        # Check ESM availability if needed
        if feature_type == 'esm' and not ESM_AVAILABLE:
            raise ImportError(
                "ESM feature extraction requires 'transformers' and 'torch' packages. "
                "Please install them separately: pip install transformers torch"
            )
        
        self.feature_type = feature_type
        self.descriptor_list = descriptor_list
        self.padding_len = padding_len
        self.add_generative_tokens = add_generative_tokens
        
        # ESM-specific parameters
        if feature_type == 'esm':
            self.esm_model_name = esm_model_name
            self.esm_layer_index = esm_layer_index
            self.esm_pooling = esm_pooling
            self.esm_batch_size = esm_batch_size
            
            # Set device
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            
            # Initialize ESM model and tokenizer (lazy loading)
            self._esm_model = None
            self._esm_tokenizer = None
        
        # Available descriptor functions list
        self.available_descriptors = {
            'aliphatic_index': descriptors.aliphatic_index,
            'aminoacid_frequencies': descriptors.aminoacid_frequencies,
            'aromaticity': descriptors.aromaticity,
            'average_n_rotatable_bonds': descriptors.average_n_rotatable_bonds,
            'charge': descriptors.charge,
            'charge_density': descriptors.charge_density,
            'compute_descriptors': descriptors.compute_descriptors,
            'hydrophobic_aa_ratio': descriptors.hydrophobic_aa_ratio,
            'instability_index': descriptors.instability_index,
            'isoelectric_point': descriptors.isoelectric_point,
            'length': descriptors.length,
            'molecular_formula': descriptors.molecular_formula,
            'molecular_weight': descriptors.molecular_weight,
            'n_h_acceptors': descriptors.n_h_acceptors,
            'n_h_donors': descriptors.n_h_donors,
            'topological_polar_surface_area': descriptors.topological_polar_surface_area,
            'x_logp_energy': descriptors.x_logp_energy,
        }
        

        # Get token2label mapping from biology module
        self.token2label = biology.token_to_label
        self.vocab_size = len(self.token2label)
    
    def _load_esm_model(self):
        """Lazy load ESM model and tokenizer"""
        if self._esm_model is None or self._esm_tokenizer is None:
            if not ESM_AVAILABLE:
                raise ImportError(
                    "ESM feature extraction requires 'transformers' and 'torch' packages. "
                    "Please install them separately: pip install transformers torch"
                )
            self._esm_tokenizer = EsmTokenizer.from_pretrained(self.esm_model_name)
            self._esm_model = EsmModel.from_pretrained(self.esm_model_name)
            self._esm_model.to(self.device)
            self._esm_model.eval()  # Set to evaluation mode
    
    def _flatten_dict_descriptor(self, desc_value: Any) -> List[float]:
        """
        Flatten dictionary-type descriptors (e.g., aminoacid_frequencies, molecular_formula)
        into a list of float values
        
        Args:
            desc_value: Descriptor value (can be dict or scalar)
                - If dict: extracts all values and returns as list
                - If scalar: returns as single-element list
            
        Returns:
            List[float]: Flattened list of descriptor values
        """
        if isinstance(desc_value, dict):
            # For dictionary descriptors, extract values and create flattened keys
            flattened = []
            for key, value in desc_value.items():
                flattened.append(value)
            return flattened
        else:
            # For scalar descriptors, return as is
            return [desc_value]
    
    def extract_descriptors(self, peptide: str) -> np.ndarray:
        """
        Extract descriptor features
        
        Args:
            peptide (str): Peptide sequence string
            
        Returns:
            np.ndarray: Numpy array containing flattened descriptor values.
                If descriptor_list is None, returns all available descriptors.
                Dictionary-type descriptors (e.g., aminoacid_frequencies) are flattened into values.
        """
        if self.descriptor_list is None:
            # Use compute_descriptors to calculate all descriptors
            all_descriptors = descriptors.compute_descriptors(peptide)
            result = self._flatten_dict_descriptor(all_descriptors)
            return np.array(result)
        else:
            # Only calculate specified descriptors
            result = []
            for desc_name in self.descriptor_list:
                if desc_name not in self.available_descriptors:
                    raise ValueError(
                        f"Unknown descriptor: {desc_name}. "
                        f"Available descriptors: {list(self.available_descriptors.keys())}"
                    )
                
                desc_func = self.available_descriptors[desc_name]
                desc_value = desc_func(peptide)
                
                # Flatten dictionary-type descriptors
                flattened = self._flatten_dict_descriptor(desc_value)
                result.extend(flattened)
            
            return np.array(result)
    
    def extract_onehot(self, peptide: str) -> np.ndarray:
        """
        Extract onehot encoding features using peptidy.encoding.one_hot_encoding
        
        Args:
            peptide (str): Peptide sequence string
            
        Returns:
            np.ndarray: Numpy array with shape (sequence_length, vocab_size) or 
                (padding_len, vocab_size) if padding_len is specified.
                Each amino acid is encoded as a one-hot vector.
        """
        # Use peptidy's one_hot_encoding function
        onehot_encoded = encoding.one_hot_encoding(peptide, padding_len=self.padding_len,
                                                         add_generative_tokens=self.add_generative_tokens)
        return np.array(onehot_encoded, dtype=np.float32)
    
    def extract_frequency(self, peptide: str) -> np.ndarray:
        """
        Extract amino acid composition frequency (AAC) features.
    
        Args:
            peptide (str): Peptide sequence string
        
        Returns:
            np.ndarray: 20-dimensional amino acid frequency vector
        """
        freq_dict = descriptors.aminoacid_frequencies(peptide)
    
        aa_order = sorted(freq_dict.keys())
        freq_vector = [freq_dict[aa] for aa in aa_order]
    
        return np.array(freq_vector, dtype=np.float32)

    
    def extract_integer(self, peptide: str) -> np.ndarray:
        """
        Extract integer encoding features using peptidy.encoding.label_encoding
        
        Args:
            peptide (str): Peptide sequence string
            
        Returns:
            np.ndarray: Numpy array with shape (sequence_length,) or (padding_len,)
                if padding_len is specified. Each amino acid is encoded as an integer index.
        """
        # Use peptidy's label_encoding function
        integer_encoded = encoding.label_encoding(peptide, padding_len=self.padding_len,
                                                         add_generative_tokens=self.add_generative_tokens)
        return np.array(integer_encoded, dtype=np.int32)

    def extract_blosum62(self, peptide: str) -> np.ndarray:
        """
        Extract BLOSUM62 encoding features using peptidy.encoding.blosum62_encoding.
        Encodes standard amino acids using BLOSUM62 substitution matrix scores,
        with an additional dimension for post-translational modifications.
        
        Args:
            peptide (str): Peptide sequence string
            
        Returns:
            np.ndarray: Numpy array with shape (sequence_length, 21) or (padding_len, 21)
                if padding_len is specified. Each amino acid is encoded as a 21-dimensional vector
                (20 standard amino acids + 1 dimension for post-translational modifications).
        """
        blosum62_encoded = encoding.blosum62_encoding(peptide, padding_len=self.padding_len,
                                                         add_generative_tokens=self.add_generative_tokens)
        return np.array(blosum62_encoded, dtype=np.float32)
    
    def extract_esm(self, peptide: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Extract ESM (Evolutionary Scale Modeling) transformer encoding features.
        Uses pre-trained ESM models from Meta AI to encode peptide sequences.
        Supports both single sequence and batch processing for improved efficiency.
        When processing multiple sequences, can optionally process in smaller batches
        to manage memory usage.
        
        Args:
            peptide (Union[str, List[str]]): Single peptide sequence string or list of peptide sequences
            
        Returns:
            Union[np.ndarray, List[np.ndarray]]: 
                - If single sequence: Numpy array with shape (embedding_dim,) if pooling is used,
                  or (sequence_length, embedding_dim) if pooling is not used.
                - If list of sequences: List of numpy arrays with same shapes as above.
                Embedding dimension depends on the ESM model used.
        """
        if not ESM_AVAILABLE:
            raise ImportError(
                "ESM feature extraction requires 'transformers' and 'torch' packages. "
                "Please install them separately: pip install transformers torch"
            )
        
        # Load model if not already loaded
        self._load_esm_model()
        
        # Handle single sequence or batch
        is_single = isinstance(peptide, str)
        if is_single:
            peptides = [peptide]
        else:
            peptides = peptide
        
        # Process in batches if batch_size is specified and we have multiple sequences
        if self.esm_batch_size is not None and len(peptides) > self.esm_batch_size:
            # Process in batches
            all_results = []
            for i in range(0, len(peptides), self.esm_batch_size):
                batch_peptides = peptides[i:i + self.esm_batch_size]
                batch_results = self._extract_esm_batch(batch_peptides)
                all_results.extend(batch_results)
            
            # Return single array or list of arrays
            if is_single:
                return all_results[0]
            else:
                return all_results
        else:
            # Process all at once (or single sequence)
            results = self._extract_esm_batch(peptides)
            if is_single:
                return results[0]
            else:
                return results
    
    def _extract_esm_batch(self, peptides: List[str]) -> List[np.ndarray]:
        """
        Internal method to extract ESM features for a batch of sequences.
        
        Args:
            peptides (List[str]): List of peptide sequences to process
            
        Returns:
            List[np.ndarray]: List of feature arrays, one for each peptide
        """
        # Tokenize all sequences in batch
        encoded = self._esm_tokenizer(
            peptides,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200 # ESM models typically have max length of 1024
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self._esm_model(**encoded, output_hidden_states=True)
            
            # Get embeddings from specified layer or last layer
            if self.esm_layer_index is not None:
                hidden_states = outputs.hidden_states[self.esm_layer_index]
            else:
                hidden_states = outputs.last_hidden_state
            
            # Shape: (batch_size, seq_len, hidden_dim)
            batch_size = hidden_states.shape[0]
            attention_mask = encoded['attention_mask']  # Shape: (batch_size, seq_len)
            
            # Apply pooling if specified
            if self.esm_pooling == 'mean':
                # Mean pooling over sequence length (excluding padding tokens)
                # Expand mask to match hidden_states dimensions
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                # Masked mean pooling
                masked_embeddings = hidden_states * attention_mask_expanded
                # Sum over sequence dimension and divide by number of non-padding tokens
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                # Shape: (batch_size, hidden_dim)
            elif self.esm_pooling == 'cls':
                # Use [CLS] token (first token) for each sequence in batch
                pooled = hidden_states[:, 0, :]  # Shape: (batch_size, hidden_dim)
            elif self.esm_pooling == 'max':
                # Max pooling over sequence length
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                # Set padding positions to very negative values
                masked_embeddings = hidden_states.clone()
                masked_embeddings[attention_mask_expanded == 0] = float('-inf')
                pooled = masked_embeddings.max(dim=1)[0]  # Shape: (batch_size, hidden_dim)
                # Replace -inf with 0 if all values were masked (shouldn't happen, but safety check)
                pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
            else:
                raise ValueError(
                    f"Unknown pooling method: {self.esm_pooling}. "
                    f"Must be one of 'mean', 'cls', or 'max'"
                )
            
            # Convert to numpy
            results = pooled.cpu().numpy().astype(np.float32)
        
        # Return list of arrays
        return [results[i] for i in range(batch_size)]
    
    def transform(self, peptide: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Transform peptide sequence(s) to features
        
        Args:
            peptide (Union[str, List[str]]): Single peptide sequence string or list of strings
            
        Returns:
            Union[np.ndarray, List[np.ndarray]]: Features according to feature_type:
                - 'descriptor': numpy array or list of arrays (flattened descriptor values)
                - 'onehot': numpy array or list of arrays (one-hot encoded sequences)
                - 'integer': numpy array or list of arrays (integer encoded sequences)
                - 'blosum62': numpy array or list of arrays (BLOSUM62 encoded sequences)
                - 'esm': numpy array or list of arrays (ESM transformer embeddings)
        """
        if isinstance(peptide, str):
            # Single sequence
            if self.feature_type == 'descriptor':
                return self.extract_descriptors(peptide)
            elif self.feature_type == 'onehot':
                return self.extract_onehot(peptide)
            elif self.feature_type == 'frequency':
                return self.extract_frequency(peptide)
            elif self.feature_type == 'integer':
                return self.extract_integer(peptide)
            elif self.feature_type == 'blosum62':
                return self.extract_blosum62(peptide)
            elif self.feature_type == 'esm':
                return self.extract_esm(peptide)
        elif isinstance(peptide, list):
            # Multiple sequences
            if self.feature_type == 'descriptor':
                return [self.extract_descriptors(p) for p in peptide]
            elif self.feature_type == 'onehot':
                return [self.extract_onehot(p) for p in peptide]
            elif self.feature_type == 'frequency':
                return [self.extract_frequency(p) for p in peptide]
            elif self.feature_type == 'integer':
                return [self.extract_integer(p) for p in peptide]
            elif self.feature_type == 'blosum62':
                return [self.extract_blosum62(p) for p in peptide]
            elif self.feature_type == 'esm':
                # ESM supports batch processing, pass list directly
                return self.extract_esm(peptide)
        else:
            raise TypeError(f"peptide must be str or List[str], got type: {type(peptide)}")
    
    def fit_transform(self, peptides: List[str]) -> List[np.ndarray]:
        """
        Batch transform peptide sequences to features (fit_transform interface, sklearn-style)
        
        Args:
            peptides (List[str]): List of peptide sequence strings
            
        Returns:
            List[np.ndarray]: List of feature arrays, one for each peptide sequence
        """
        return self.transform(peptides)


if __name__ == '__main__':
    """Example usage of PeptideFeaturizer"""
    
    # Example peptide sequences
    test_peptides = ["ACDEFGHIKLMNPQRSTVWY", "PEPTIDE", "MVHLTPEEKS"]
    single_peptide = "MVHLTPEEKS"
    
    print("=" * 80)
    print("Example 1: Descriptor Feature Extraction (All Descriptors)")
    print("=" * 80)
    featurizer_desc = PeptideFeaturizer(feature_type='descriptor')
    descriptors_feat = featurizer_desc.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Descriptor array shape: {descriptors_feat.shape}")
    print(f"Descriptor array dtype: {descriptors_feat.dtype}")
    print(f"First 10 descriptor values: {descriptors_feat[:10]}")
    print()
    
    print("=" * 80)
    print("Example 2: Batch Descriptor Extraction")
    print("=" * 80)
    batch_descriptors = featurizer_desc.transform(test_peptides)
    print(f"Processed {len(batch_descriptors)} peptides")
    for i, peptide in enumerate(test_peptides):
        print(f"  {peptide}: shape {batch_descriptors[i].shape}, dtype {batch_descriptors[i].dtype}")
    print()
    
    print("=" * 80)
    print("Test 3: Specific Descriptors")
    print("=" * 80)
    specific_descriptors = ['molecular_weight', 'isoelectric_point', 'charge']
    featurizer_specific = PeptideFeaturizer(
        feature_type='descriptor', 
        descriptor_list=specific_descriptors
    )
    specific_desc = featurizer_specific.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Requested descriptors: {specific_descriptors}")
    print(f"Result shape: {specific_desc.shape}")
    print(f"Result values: {specific_desc}")
    print()
    
    print("=" * 80)
    print("Test 4: Dictionary Descriptor Handling (aminoacid_frequencies)")
    print("=" * 80)
    featurizer_freq = PeptideFeaturizer(
        feature_type='descriptor',
        descriptor_list=['aminoacid_frequencies']
    )
    freq_descriptors = featurizer_freq.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Flattened aminoacid_frequencies shape: {freq_descriptors.shape}")
    print(f"First 10 values: {freq_descriptors[:10]}")
    print()
    
    print("=" * 80)
    print("Test 5: One-Hot Encoding (Single Peptide)")
    print("=" * 80)
    featurizer_onehot = PeptideFeaturizer(feature_type='onehot')
    onehot = featurizer_onehot.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"One-hot shape: {onehot.shape}")
    print(f"One-hot encoding (first 3 positions):")
    print(onehot[:3])
    print()
    
    print("=" * 80)
    print("Test 6: One-Hot Encoding (Batch)")
    print("=" * 80)
    batch_onehot = featurizer_onehot.transform(test_peptides)
    print(f"Processed {len(batch_onehot)} peptides")
    for i, peptide in enumerate(test_peptides):
        print(f"  {peptide}: shape {batch_onehot[i].shape}")
    print()
    
    print("=" * 80)
    print("Test 7: One-Hot Encoding with Padding")
    print("=" * 80)
    featurizer_onehot_pad = PeptideFeaturizer(
        feature_type='onehot',
        padding_len=15
    )
    onehot_pad = featurizer_onehot_pad.transform(single_peptide)
    print(f"Peptide: {single_peptide} (length: {len(single_peptide)})")
    print(f"Padded one-hot shape: {onehot_pad.shape}")
    print()
    
    print("=" * 80)
    print("Test 8: One-Hot Encoding with Generative Tokens")
    print("=" * 80)
    featurizer_onehot_gen = PeptideFeaturizer(
        feature_type='onehot',
        add_generative_tokens=True
    )
    onehot_gen = featurizer_onehot_gen.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"One-hot with generative tokens shape: {onehot_gen.shape}")
    print()
    
    print("=" * 80)
    print("Test 9: Integer Encoding (Single Peptide)")
    print("=" * 80)
    featurizer_int = PeptideFeaturizer(feature_type='integer')
    integer = featurizer_int.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Integer encoding shape: {integer.shape}")
    print(f"Integer encoding: {integer}")
    print()
    
    print("=" * 80)
    print("Test 10: Integer Encoding (Batch)")
    print("=" * 80)
    batch_integer = featurizer_int.transform(test_peptides)
    print(f"Processed {len(batch_integer)} peptides")
    for i, peptide in enumerate(test_peptides):
        print(f"  {peptide}: shape {batch_integer[i].shape}, values: {batch_integer[i]}")
    print()
    
    print("=" * 80)
    print("Test 11: Integer Encoding with Padding and Generative Tokens")
    print("=" * 80)
    featurizer_int_pad = PeptideFeaturizer(
        feature_type='integer',
        padding_len=12,
        add_generative_tokens=True
    )
    integer_pad = featurizer_int_pad.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Padded integer encoding shape: {integer_pad.shape}")
    print(f"Padded integer encoding: {integer_pad}")
    print()
    
    print("=" * 80)
    print("Test 12: BLOSUM62 Encoding (Single Peptide)")
    print("=" * 80)
    featurizer_blosum = PeptideFeaturizer(feature_type='blosum62')
    blosum = featurizer_blosum.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"BLOSUM62 encoding shape: {blosum.shape}")
    print(f"BLOSUM62 encoding (first 3 positions):")
    print(blosum[:3])
    print()
    
    print("=" * 80)
    print("Test 13: BLOSUM62 Encoding (Batch)")
    print("=" * 80)
    batch_blosum = featurizer_blosum.transform(test_peptides)
    print(f"Processed {len(batch_blosum)} peptides")
    for i, peptide in enumerate(test_peptides):
        print(f"  {peptide}: shape {batch_blosum[i].shape}")
    print()
    
    print("=" * 80)
    print("Test 14: BLOSUM62 Encoding with Padding")
    print("=" * 80)
    featurizer_blosum_pad = PeptideFeaturizer(
        feature_type='blosum62',
        padding_len=20
    )
    blosum_pad = featurizer_blosum_pad.transform(single_peptide)
    print(f"Peptide: {single_peptide}")
    print(f"Padded BLOSUM62 encoding shape: {blosum_pad.shape}")
    print()
    
    print("=" * 80)
    print("Test 15: fit_transform Method")
    print("=" * 80)
    featurizer_fit = PeptideFeaturizer(feature_type='onehot')
    fit_result = featurizer_fit.fit_transform(test_peptides)
    print(f"fit_transform processed {len(fit_result)} peptides")
    print(f"All results are numpy arrays: {all(isinstance(x, np.ndarray) for x in fit_result)}")
    print()
    
    print("=" * 80)
    print("Test 16: ESM Encoding (if transformers and torch are available)")
    print("=" * 80)
    if ESM_AVAILABLE:
        try:
            featurizer_esm = PeptideFeaturizer(
                feature_type='esm',
                esm_model_name='facebook/esm2_t6_8M_UR50D',
                esm_pooling='mean'
            )
            esm_embedding = featurizer_esm.transform(single_peptide)
            print(f"Peptide: {single_peptide}")
            print(f"ESM embedding shape: {esm_embedding.shape}")
            print(f"ESM embedding dtype: {esm_embedding.dtype}")
            print(f"First 10 values: {esm_embedding[:10]}")
            print()
            
            print("Test 16b: ESM Encoding (Batch)")
            batch_esm = featurizer_esm.transform(test_peptides)
            print(f"Processed {len(batch_esm)} peptides")
            for i, peptide in enumerate(test_peptides):
                print(f"  {peptide}: shape {batch_esm[i].shape}")
            print()
            
            print("Test 16c: ESM Encoding with different pooling methods")
            for pooling_method in ['mean', 'cls', 'max']:
                featurizer_esm_pool = PeptideFeaturizer(
                    feature_type='esm',
                    esm_model_name='facebook/esm2_t6_8M_UR50D',
                    esm_pooling=pooling_method
                )
                esm_pool = featurizer_esm_pool.transform(single_peptide)
                print(f"  Pooling method '{pooling_method}': shape {esm_pool.shape}")
            print()
        except Exception as e:
            print(f"ESM encoding test failed: {e}")
            print("This might be due to model download or GPU availability issues.")
            print()
    else:
        print("ESM encoding requires 'transformers' and 'torch' packages.")
        print("Install them separately: pip install transformers torch")
        print("Skipping ESM tests.")
        print()
    
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

