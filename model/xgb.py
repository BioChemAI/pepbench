from xgboost import XGBClassifier, XGBRegressor
from .base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, task='classification', random_state=42, **kwargs):
        super().__init__()
        self.task = task
        if task == 'classification':
            self.model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1,  # 控制多线程以减少随机性
                **kwargs
            )
        elif task == 'regression':
            self.model = XGBRegressor(
                random_state=random_state,
                n_jobs=1,
                **kwargs
            )
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]  # 概率，用于计算 AUC 等
        else:  # regression
            return self.model.predict(X)  # 连续值


    def evaluate(self, X, y):
        return self.model.score(X, y)  # 分类：accuracy，回归：R²
