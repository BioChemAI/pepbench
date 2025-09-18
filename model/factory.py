from .rf import RandomForestModel
from .svm import SVMModel
from .xgb import XGBoostModel
from .transformer import TransformerModel
from .lstm import LSTMModel
from .esm_model import ESMModel
from .pepbert import PepBERTModel

def build_model(name, task, input_dim=None, max_len=50, device=None, model_path=None, random_state=42, **kwargs):
    """
    构建模型对象。

    参数:
        name: 模型名称（如 'rf', 'svm', 'transformer' 等）
        task: 'classification' 或 'regression'
        input_dim: 每个输入样本的特征向量
        max_len: 输入序列最大长度
        device: 用于深度模型的计算设备（如 'cuda' 或 'cpu'）
        random_state: 随机种子，用于传统模型和接口统一

    返回:
        已实例化的模型对象
    """
    name = name.lower()

    if model_path is None:
        if name == "pepbert":
            model_path = "MODEL/prot_bert"
        elif name == "esm":
            model_path = "MODEL/esm2_t12_35M_UR50D"

    if name == 'rf':
        return RandomForestModel(task=task, random_state=random_state, **kwargs)

    elif name == 'svm':
        return SVMModel(task=task, random_state=random_state, **kwargs)

    elif name == 'xgb':
        return XGBoostModel(task=task, random_state=random_state, **kwargs)

    elif name == 'transformer':
        return TransformerModel(task=task, device = device, max_len=max_len)
    
    elif name == 'lstm':
        return LSTMModel(task=task, device=device, max_len = max_len)

    elif name == "pepbert":
        return PepBERTModel(model_path=model_path, device=device, max_len=max_len)

    elif name == "esm":
        return ESMModel(model_path=model_path, device=device, max_len=max_len)

    else:
        raise ValueError(f"Unsupported model name: {name}")
