import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, max_len=50, task='classification', device=None):
        super().__init__()
        self.task = task
        self.device = device or torch.device('cpu')

        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(128, 1)

        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # 简单池化
        out = self.classifier(x)
        return out

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.forward(X)
            if self.task == 'classification':
                return output.argmax(dim=1).cpu().numpy()
            else:
                return output.squeeze(1).cpu().numpy()
