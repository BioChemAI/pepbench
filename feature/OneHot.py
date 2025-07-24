
# import numpy as np

# class OneHotEncoder:
#     def __init__(self, max_len=50, flatten=True):
#         self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#         self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
#         self.unknown_index = 20  # 第21类（未知氨基酸）
#         self.max_len = max_len   # 固定序列长度
#         self.flatten = flatten   # 是否展开为一维向量

#     def encode(self, sequence):
#         sequence = sequence[:self.max_len]  # 截断过长序列
#         encoding = np.zeros((self.max_len, 21), dtype=np.float32)
#         for i, aa in enumerate(sequence):
#             idx = self.aa_to_idx.get(aa, self.unknown_index)
#             encoding[i][idx] = 1.0

#         if self.flatten:
#             return encoding.flatten()  # shape: (max_len × 21,)
#         else:
#             return encoding            # shape: (max_len, 21)


import numpy as np

class OneHotEncoder:
    def __init__(self, max_len=50, flatten=True):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.max_len = max_len
        self.flatten = flatten

    def encode(self, sequence):
        encoding = np.zeros((self.max_len, 20), dtype=np.float32)

        for i in range(min(len(sequence), self.max_len)):
            aa = sequence[i]
            if aa in self.aa_to_idx:
                idx = self.aa_to_idx[aa]
                encoding[i][idx] = 1.0
            # 如果未知氨基酸，整行为0，可跳过

        if self.flatten:
            return encoding.flatten()  # shape: (max_len * 20,)
        else:
            return encoding            # shape: (max_len, 20)
