'''
for rf/svm/xgb
fun:aas--->[0,1,0,...,0]s
dim:max_len*20, numpy array
'''
import numpy as np
class OneHotEncoder:
    def __init__(self, max_len=50, flatten=True):
        self.max_len = max_len
        self.flatten = flatten

        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        self.dim = 20  # one-hot encode dim:20/21
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.aas)}

    def encode(self, sequences):
        # sequence = sequence.upper()
        n_samples = len(sequences)
        encoded_list = []
        
        for seq in sequences:
            # (max_len,20) for a sequence
            encoded_vector = np.zeros((self.max_len, self.dim), dtype=np.float32)
            for i in range(min(len(seq), self.max_len)):
                aa = seq[i]
                if aa in self.aa_to_idx:
                    idx = self.aa_to_idx[aa]
                    encoded_vector[i][idx] = 1.0
            encoded_list.append(encoded_vector)

        batch_encoded = np.stack(encoded_list)           # shape: (n_samples, max_len, 20)

        if self.flatten:
            return batch_encoded.reshape(n_samples, -1)  # shape: (n_samples, max_len * 20,)
        else:
            return batch_encoded                         # shape: (n_samples, max_len, 20)
