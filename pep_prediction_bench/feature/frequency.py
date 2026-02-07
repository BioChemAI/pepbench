'''
For RF/SVM/XGB
Function: AAC (Amino Acid Composition)
Format: aas ---> [freq_A, freq_C, ..., freq_Y]  (20-dim vector per sequence)
Dim: 20, numpy array
'''

import numpy as np

class AACEncoder:
    def __init__(self):
        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        self.dim = len(self.aas)  # 20

    def encode(self, sequences):
        """
        Encode sequences into Amino Acid Composition (AAC) feature vectors.

        Parameters:
            sequences (list of str): List of amino acid sequences.

        Returns:
            np.ndarray: Shape (n_samples, 20), each row is the AAC of a sequence.
        """
        X = []
        for seq in sequences:
            # Clean sequence: uppercase and keep only standard AAs
            clean_seq = ''.join(aa for aa in seq.upper() if aa in self.aas)
            length = len(clean_seq)
            
            if length == 0:
                # If no valid amino acids, return zero vector
                comp = [0.0] * self.dim
            else:
                comp = [clean_seq.count(aa) / length for aa in self.aas]
            
            X.append(comp)
        
        return np.array(X, dtype=np.float32)  # shape: (n_samples, 20)
