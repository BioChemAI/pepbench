'''
for lstm/transformer
fun:aas-->numbers
type:numpy array
'''


import numpy as np
class IntegerEncoder:
    def __init__(self, max_len=50):
        self.max_len = max_len

        self.aa_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                        'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, }
        
        self.padding_idx = 0   # padding: 0
        self.vocab_size = 22   # 0~21

    def encode(self, sequences):
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded_list = []
        for seq in sequences:
            seq = seq.strip().upper()
            encoded_vector = [
                self.aa_to_int.get(aa, self.padding_idx)
                for aa in seq
            ]
            
            if len(encoded_vector) > self.max_len:
                encoded_vector = encoded_vector[:self.max_len]
            else:
                encoded_vector = encoded_vector + [self.padding_idx] * (self.max_len - len(encoded_vector))
            
            encoded_list.append(encoded_vector)
        
        return np.array(encoded_list, dtype=np.int64)
