from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, task='classification', 
                 n_estimators=200,      # 树的数量，可以适当增加
                 max_depth=None,        # 限制树深度，例如 10/20，None 表示不限制
                 min_samples_split=5,   # 分裂所需最小样本数（默认2，调大防止过拟合）
                 min_samples_leaf=2,    # 叶子最小样本数
                 max_features="sqrt",   # 每次分裂使用的特征数（分类推荐 "sqrt"，回归推荐 "auto" 或 "sqrt"）
                 random_state=42):
        super().__init__()
        self.task = task
        if task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1  # 并行加速
            )
        elif task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]  # 概率，用于计算 AUC 等
        else:  # regression
            return self.model.predict(X)  # 连续值

    def evaluate(self, X, y):
        return self.model.score(X, y)  # 分类返回 accuracy，回归返回 R²
