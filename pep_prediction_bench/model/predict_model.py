import torch
import torch.nn as nn

class PredictModel(nn.Module):
    def __init__(self,hidden_size, task):
        super().__init__()
        if task == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus()
            )
    def forward(self,embedding):
        out = self.output_layer(embedding)
        return out
