# model/factory.py

from .rf import RandomForestModel
from .svm import SVMModel
from .logistic import LogisticRegressionModel
from .linear import LinearRegressionModel
from .xgb import XGBoostModel
from .transformer import TransformerModel
from .lstm import LSTMModel
from .esm_model import ESMModel
from .pepbert import PepBERTModel

def build_model(name, task, input_dim=None, max_len=50, device=None, model_path=None):
    """
    构建模型对象。

    参数:
        name: 模型名称（如 'rf', 'svm', 'transformer' 等）
        task: 'classification' 或 'regression'
        input_dim: 对于深度学习模型，输入维度（如 one-hot 后维度）
        max_len: 输入序列最大长度，仅用于 Transformer
        device: 用于深度模型的计算设备（如 'cuda' 或 'cpu'）

    返回:
        已实例化的模型对象
    """
    name = name.lower()

    if name == 'rf':
        return RandomForestModel(task=task)

    elif name == 'svm':
        return SVMModel(task=task)

    elif name == 'logistic':
        return LogisticRegressionModel(task=task)

    elif name == 'linear':
        return LinearRegressionModel(task=task)

    elif name == 'xgb':
        return XGBoostModel(task=task)

    elif name == 'transformer':
        return TransformerModel(input_dim=input_dim, max_len=max_len, task=task, device=device)
    
    elif name == 'lstm':
        return LSTMModel(input_dim=input_dim, task=task, device=device)

    elif name == 'esm':
        return ESMModel(model_path=model_path, task=task, device=device)

    elif name == 'pepbert':
        return PepBERTModel(model_path=model_path, task=task, device=device)

    else:
        raise ValueError(f"Unsupported model name: {name}")
