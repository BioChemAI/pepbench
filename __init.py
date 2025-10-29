# __init__.py
from .dataset import PepDataset, download_data
from .model_manager import ModelManager
from .data_split.random_split import random_split_dataset, similar_split_dataset
from .quick_start import compute_metrics

__version__ = "0.1.0"

__all__ = [
    "PepDataset",
    "download_data",
    "ModelManager",
    "random_split_dataset",
    "similar_split_dataset",
    "compute_metrics",
]