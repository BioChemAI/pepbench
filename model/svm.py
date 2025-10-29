# model/svm.py
from sklearn.svm import SVC, SVR
from .base import BaseModel

class SVMModel(BaseModel):
    def __init__(self, task='classification',
                 C=1.0,
                 kernel='rbf',
                 random_state=42,
                 **kwargs):
        super().__init__()
        self.task = task

        if task == 'classification':
            # output probability: probability=True
            self.model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
        elif task == 'regression':
            self.model = SVR(C=C, kernel=kernel, random_state=random_state)
        else:
            raise ValueError("--Task must be 'classification' or 'regression'.--")

    # define the training method
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # define the prediction method
    def predict(self, X):
        if self.task == 'classification':
            return self.model.predict_proba(X)[:, 1]  # probability
        else:
            return self.model.predict(X)

    # define the evaluation method
    def evaluate(self, X, y):
        return self.model.score(X, y)  # classification-> accuracy，regression-> R²
