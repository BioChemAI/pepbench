import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        raise NotImplementedError("train() method must be implemented.")

    def predict(self, X):
        raise NotImplementedError("predict() method must be implemented.")

    def evaluate(self, X, y):
        raise NotImplementedError("evaluate() method must be implemented.")
