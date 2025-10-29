# model/xgb.py
from xgboost import XGBClassifier, XGBRegressor
from .base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, task='classification',
                 random_state=42,
                 **kwargs):
        super().__init__()
        self.task = task
        if task == 'classification':
            self.model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        elif task == 'regression':
            self.model = XGBRegressor(
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            raise ValueError("--Task must be either 'classification' or 'regression'.--")

    # define the training method
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # define the prediction method
    def predict(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]
        else:  # regression
            return self.model.predict(X)

    # define the evaluation method
    def evaluate(self, X, y):
        return self.model.score(X, y)  # classification-> accuracy，regression-> R²
