# model/rf.py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, task='classification',
                 n_estimators=200,      # Number of trees: can be moderately increased
                 max_depth=None,        # Maximum depth of the trees (e.g., 10 or 20): None means no restriction
                 min_samples_split=5,   # Minimum number of samples required to split an internal node (default is 2)
                 min_samples_leaf=2,    # Minimum number of samples required to be at a leaf node
                 max_features="sqrt",   # Number of features to consider when looking for the best split ("sqrt" recommended for classification; "auto" or "sqrt" for regression)
                 random_state=42,
                 **kwargs):
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
                n_jobs=-1
            )
        elif task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                bootstrap=True,
                n_jobs=-1
            )
        else:
            raise ValueError("--Task must be 'classification' or 'regression'.--")

    # define the training method
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # define the prediction method
    def predict(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]  # probability
        else:  # regression
            return self.model.predict(X)

    # define the evaluation method
    def evaluate(self, X, y):
        return self.model.score(X, y)  # classification-> accuracy，regression-> R²
