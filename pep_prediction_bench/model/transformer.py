import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """Standard sine positional encoding, adapted to (batch, seq_len, d_model)"""
    def __init__(self, d_model, dropout, max_len=198):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *(-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Classification/regression model based on Transformer Encoder"""
    def __init__(self, task=None, device = None, max_len= 20, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.task = task
        self.device = device if device is not None else torch.device('cpu')
        self.max_len = max_len
        self.d_model = d_model
        # Size of the amino acid dictionary: 20 natural amino acids + 1 (padding_idx=0)
        self.vocab_size = 21
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        # nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.max_len)
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     nn.init.normal_(self.embedding.weight,mean=0,std=self.d_model**-0.5)
    #     if self.embedding.padding_idx is not None:
    #         nn.init.constant_(self.embedding.weight[self.embedding.padding_idx],0)
    #     nn.init.xavier_uniform_(self.fc.weight)
    #     nn.init.constant_(self.fc.bias,0)


    def forward(self, src):
        """src: (batch, seq_len), a sequence after integer encoding, where 0 represents padding"""
        if src.device != next(self.parameters()).device:
            src = src.to(next(self.parameters()).device)
        # mask: True indicates padding and needs to be ignored
        mask = (src == 0)
        # Embedding + Scaling
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        # src_embedded = self.embedding(src)
        # Embedding + Scaling
        src_embedded = self.pos_encoder(src_embedded)
        # Transformer encoder
        output = self.transformer_encoder(src_embedded, src_key_padding_mask=mask) #b,l,d

        output = output[:,0,:]

        # masked average pooling
        # lengths = (~mask).sum(dim=1, keepdim=True)  # The effective length of each sequence
        # output = output.masked_fill(mask.unsqueeze(-1), 0.0)  # Set the padding position to zero
        # output = output.sum(dim=1) / lengths  # (batch, d_model)
        # Dropout and fully connected
        # output = self.dropout(output)
        output = self.fc(output)
        # Choose the activation function according to the task
        if self.task == 'classification':
            output = torch.sigmoid(output)

        return output

