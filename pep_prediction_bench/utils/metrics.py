"""
Evaluation utilities for classification and regression tasks.
This module provides standardized functions to compute common evaluation metrics
for both classification and regression problems.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Compute common classification metrics.

    Parameters
    ----------
    y_true : Ground truth target values.
    y_pred : Estimated targets as returned by a classifier.
    y_prob : optional, Predicted probabilities.

    Returns
    -------
    metrics : dict
        Dictionary containing the following metrics (when applicable):
        - 'accuracy': Accuracy score
        - 'f1_score': Macro-averaged F1 score
        - 'precision': Macro-averaged precision
        - 'recall': Macro-averaged recall
        - 'auc': ROC AUC score (only for binary classification with y_prob provided)

    Notes
    -----
    - Macro averaging is used for multiclass metrics.
    - AUC is only computed if `y_prob` is provided and the problem is binary.
    - If AUC computation fails (e.g., due to constant predictions), it is omitted with a warning.
    """
    # Core metrics
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

    # Only for binary classification
    if y_prob is not None and len(set(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics['auc'] = auc
        except:
            pass
    return metrics

def evaluate_regression(y_true, y_pred):
    """
    Compute common regression metrics.

    Parameters
    ----------
    y_true : Ground truth target values.
    y_pred : Estimated targets as returned by a regressor.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'mape': Mean Absolute Percentage Error (in decimal, e.g., 0.1 = 10%)
        - 'r2_score': R² (coefficient of determination)

    Notes
    -----
    - MAPE may be infinite or undefined if any y_true value is zero.
      In such cases, scikit-learn returns `inf` or `nan`, which is preserved here.
    """
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
