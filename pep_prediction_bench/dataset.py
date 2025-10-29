"""
Custom Dataset for peptide sequence data.

Supports multiple feature encoding strategies based on model type:
- 'ml': traditional machine learning (e.g., one-hot, descriptors)
- 'dl': deep learning model
- 'll': language model (raw sequences, e.g., esm, pepbert)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import urllib.request
from pathlib import Path
from urllib.error import URLError
import shutil

from feature.onehot import OneHotEncoder
from feature.descriptor import PeptidyDescriptorEncoder
from feature.integer import IntegerEncoder

class PepDataset(Dataset):
    """
    A Dataset for peptide sequences with flexible feature encoding.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing peptide data. Must have columns: 'peps', 'label'.
    max_len : int, default=50
        Maximum sequence length for padding/truncation.
    task : {'classification', 'regression'}, default='classification'
        Type of learning task.
    feature_type : {'onehot', 'descriptor'}, default='onehot'
        Feature encoding method (only used when model_type='ml').
    model_type : {'ml', 'dl', 'll'}, default='ml'
        Model category:
        - 'ml': returns flattened numerical features (float32)
        - 'dl': returns integer token sequences (long)
        - 'll': returns raw string sequences

    Attributes
    ----------
    data : pd.DataFrame
        Raw loaded data.
    sequences : np.ndarray
        Peptide sequences as strings.
    labels : np.ndarray
        Labels, typed according to task.
    features : Union[np.ndarray, List[str]]
        Encoded features or raw sequences.
    """
    def __init__(self, csv_path, max_len = 50, task='Classification', feature_type='onehot', model_type='ml'):
        # Data load
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"--Dataset not found--: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.sequences = self.data['peps'].values  # pep sequence
        self.labels = self.data['label'].values.astype(np.float32 if task == 'regression' else int)  # pep label
        self.feature_type = feature_type
        self.model_type = model_type

        # feature encoder
        if model_type == 'ml':
            if feature_type == 'onehot':
                encoder = OneHotEncoder(max_len=max_len, flatten=True)
                self.features = encoder.encode(self.sequences)
            elif feature_type == 'descriptor':
                encoder = PeptidyDescriptorEncoder()
                self.features = encoder.encode(self.sequences)
            else:
                raise ValueError("--feature_type must be one of ['onehot', 'descriptor'] for traditional models.--")
        elif model_type == 'dl':
            encoder = IntegerEncoder(max_len=max_len)
            self.features = encoder.encode(self.sequences)
        elif model_type == 'll':
            self.features = self.sequences
        else:
            raise ValueError("--Unknown model type.--")

    def __len__(self):
            return len(self.sequences)
    
    def _get_label_tensor(self, idx):
        """Helper to convert label to appropriate tensor type."""
        label_val = self.labels[idx]
        if self.task == "classification":
            return torch.tensor(label_val, dtype=torch.long)
        else:
            return torch.tensor(label_val, dtype=torch.float32)

    def __getitem__(self, idx):
        label = self._get_label_tensor(idx)
        if self.model_type == 'll':
            # features[idx]:string
            return self.features[idx], self.labels[idx]
        elif self.model_type == 'dl':
            feature = torch.tensor(self.features[idx], dtype=torch.long)
            return feature, label             
        else:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            return feature, label
    
def download_data(filename, force=False, return_dataframe=False):
    """
    Download a dataset file from remote and save to ./data_download/.

    Parameters
    ----------
    filename : str
        Name of the file to download (e.g., "pep_classification.csv").
    force : bool
        If True, re-download even if exists.
    return_dataframe : bool
        If True, return a pandas DataFrame.

    Returns
    -------
    str or pd.DataFrame
        Path to file or DataFrame.
    """
    RECORD_ID = "17455060"
    BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files/"

    url = BASE_URL + filename + "?download=1"

    download_dir = Path("data_download")
    download_dir.mkdir(exist_ok=True)
    local_path = download_dir / filename

    if not local_path.exists() or force:
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, str(local_path))
            print(f"Saved to: {local_path.resolve()}")
        except URLError as e:
            raise RuntimeError(
                f"Failed to download '{filename}' from {url}. "
                f"Check filename and internet."
            ) from e
    else:
        print(f"File already exists: {local_path.resolve()}")

    if return_dataframe:
        return pd.read_csv(local_path)
    else:
        return str(local_path.resolve())
