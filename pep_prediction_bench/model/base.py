# model/base.py
# ======================================================
# Base model class for the project
# This class serves as the parent of all models and defines
# the unified interface:
# - fit: train the model
# - predict: make predictions
# - evaluate: evaluate model performance
# Any model inheriting from BaseModel must implement these methods
# ======================================================

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
