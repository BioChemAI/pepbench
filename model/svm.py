
# from sklearn.svm import LinearSVC, LinearSVR
# from .base import BaseModel

# class SVMModel(BaseModel):
#     def __init__(self, task='classification', C=1.0, random_state=42):
#         super().__init__()
#         self.task = task
#         if task == 'classification':
#             # LinearSVC 不支持 probability=True（只能 decision_function）
#             self.model = LinearSVC(C=C, random_state=random_state)
#         elif task == 'regression':
#             self.model = LinearSVR(C=C, random_state=random_state)
#         else:
#             raise ValueError("Task must be 'classification' or 'regression'")

#     def fit(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X):
#         return self.model.predict(X)

#     def evaluate(self, X, y):
#         return self.model.score(X, y)



# svm.py
from sklearn.svm import SVC, SVR
from .base import BaseModel

class SVMModel(BaseModel):
    def __init__(self, task='classification', C=1.0, kernel='rbf', random_state=42):
        super().__init__()
        self.task = task

        if task == 'classification':
            # 正宗 SVC，可输出概率（需要 probability=True）
            self.model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
        elif task == 'regression':
            self.model = SVR(C=C, kernel=kernel)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        # 分类返回概率，回归返回预测值
        if self.task == 'classification':
            return self.model.predict_proba(X)[:, 1]  # 返回正类概率
        else:
            return self.model.predict(X)

    def evaluate(self, X, y):
        # 分类用准确率，回归用 R^2
        return self.model.score(X, y)
