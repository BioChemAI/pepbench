from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, task='classification', n_estimators=100, random_state=42):
        self.task = task
        if task == 'classification':
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        elif task == 'regression':
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)  # 分类返回accuracy，回归返回R^2
