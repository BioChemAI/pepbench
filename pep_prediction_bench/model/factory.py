# model/factory.py
# ===============================
# This function is the unified model construction factory in this project.
# It supports both classical ML models and deep learning models.
# Whether RandomForest, SVM, XGBoost, Transformer, LSTM, ESM, or PepBERT,
# all models are instantiated through this interface to guarantee consistent usage.
# ===============================
from .rf import RandomForestModel
from .svm import SVMModel
from .xgb import XGBoostModel
from .transformer import TransformerModel
from .lstm import LSTMModel
from .esm_model import ESMModel
from .pepbert import PepBERTModel


def build_model(name, task, max_len=50, device=None, model_path=None, random_state=42, **kwargs):
    """
    Build model object.

    Args:
        name: model name (e.g., 'rf', 'svm', 'transformer', etc.)
        task: 'classification' or 'regression'
        input_dim: feature vector dimension of each input sample
        max_len: maximum input sequence length
        device: computation device for deep models (e.g., 'cuda' or 'cpu')
        random_state: random seed for classical ML models

    Returns:
        Instantiated model object
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
        return TransformerModel(task=task, device=device, max_len=max_len)

    elif name == 'lstm':
        return LSTMModel(task=task, device=device, max_len=max_len)

    elif name == "pepbert":
        return PepBERTModel(model_path=model_path, device=device, max_len=max_len)

    elif name == "esm":
        return ESMModel(model_path=model_path, device=device, max_len=max_len)

    else:
        raise ValueError(f"Unsupported model name: {name}")
