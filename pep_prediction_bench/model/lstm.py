# model/lstm.py

import torch
import torch.nn as nn
import random
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, embedding_dim=50, hidden_dim=256, num_layers=2, task='classification',
            dropout=0.2, device=None, max_len = None):
        super().__init__()
        self.task = task
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.vocab_size = 21
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Assume that 0 is a padding token.
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.to(self.device)

    def _process_sequences(self, sequences):
        """Processing the sequence: truncate or pad to the max_len length"""
        processed = []
        for seq in sequences:
            if len(seq) > self.max_len:
                # Truncate the part that exceeds max_len
                processed_seq = seq[:self.max_len]
            elif len(seq) < self.max_len:
                # Pad to the max_len length with 0s.
                processed_seq = seq + [0] * (self.max_len - len(seq))
            else:
                processed_seq = seq
            processed.append(processed_seq)
        return np.array(processed)
    def forward(self, x):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        # x should be an integer index with a shape of (B, L)
        x_embedded = self.embedding(x)  # (B, L, embedding_dim)
        lstm_out, _ = self.lstm(x_embedded)  # (B, L, 2*H)
        pooled = lstm_out.mean(dim=1)   # (B, 2*H)
        out = self.classifier(pooled)   # (B, 1)
        if self.task == 'classification':
            out = torch.sigmoid(out)
        return out
