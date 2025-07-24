from sklearn.svm import SVC, SVR
from .base import BaseModel

class SVMModel(BaseModel):
    def __init__(self, task='classification', kernel='rbf', C=1.0):
        self.task = task
        if task == 'classification':
            self.model = SVC(kernel=kernel, C=C, probability=True)  # 支持 predict_proba
        elif task == 'regression':
            self.model = SVR(kernel=kernel, C=C)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)
