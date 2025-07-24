from sklearn.model_selection import train_test_split

class RandomSplitter:
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Train/Val/Test比例必须加和为1"
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        # 1. 划分出测试集（10%）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # 2. 在剩下的 90% 中划出验证集（比例为 0.1 / 0.9 ≈ 11.1%）
        val_ratio = self.val_size / (self.train_size + self.val_size)  # = 0.1 / 0.9
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
