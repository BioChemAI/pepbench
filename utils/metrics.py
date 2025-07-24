# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# def evaluate_classification(y_true, y_pred):
#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')
#     return {'accuracy': acc, 'f1_score': f1}

# def evaluate_regression(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return {'mse': mse, 'r2_score': r2}


from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    y_true: 真实标签
    y_pred: 预测标签（整型）
    y_prob: 可选，预测概率（float），仅用于计算 AUC（二分类）
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

    # 仅限二分类
    if y_prob is not None and len(set(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics['auc'] = auc
        except:
            pass  # AUC 可能报错

    return metrics

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2_score': r2
    }
