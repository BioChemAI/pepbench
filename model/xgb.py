from xgboost import XGBClassifier, XGBRegressor
from .base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, task='classification', **kwargs):
        self.task = task
        if task == 'classification':
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
        elif task == 'regression':
            self.model = XGBRegressor(**kwargs)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)  # 分类：accuracy，回归：R²
