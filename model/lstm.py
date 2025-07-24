# model/lstm.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=2, task='classification', dropout=0.1, device=None):
        super().__init__()
        self.task = task
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        # ✅ classification 也用 1 输出，配合 BCEWithLogitsLoss
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.to(self.device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)              # (B, L, 2*H)
        pooled = lstm_out.mean(dim=1)           # (B, 2*H)
        out = self.classifier(pooled)           # (B, 1)
        return out

    def fit(self, X_train, y_train, lr=1e-3, epochs=20, batch_size=32):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)  # (B, 1)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss() if self.task == 'classification' else nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]

                optimizer.zero_grad()
                out = self.forward(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / len(X)
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.forward(X)  # logits
            if self.task == 'classification':
                probs = torch.sigmoid(out)
                return (probs > 0.5).long().squeeze(1).cpu().numpy()
            else:
                return out.squeeze(1).cpu().numpy()
