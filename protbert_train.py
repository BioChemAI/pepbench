#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train and evaluate a frozen ProtBERT encoder with an MLP head.

The script uses a fixed learning rate and a consistent feature extraction,
validation, refit, and test procedure across datasets.
"""


import os
import json
import csv
import time
import math
import random
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any


import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from transformers import BertTokenizer, BertModel


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


from pephub.results import write_metrics_report
from pephub.splitter import split_dataset, split_dataset_by_similarity


# ==========================================================
# Global configuration
# ==========================================================


DEFAULT_LEARNING_RATE = 1e-3


DEFAULT_SEEDS = [42, 43, 44]


CLASSIFICATION_TASKS = [
    "classification",
]


REGRESSION_TASKS = [
    "regression",
]


def torch_load_compatible(path, map_location):
    """Load trusted local checkpoints across PyTorch 2.6 and older versions."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


# ==========================================================
# Random seed
# ==========================================================


def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Device
# ==========================================================


def get_device(device):

    if device.startswith("cuda") and torch.cuda.is_available():

        return torch.device(device)

    return torch.device("cpu")


# ==========================================================
# ProtBERT Encoder
# ==========================================================


class ProtBERTEncoder(nn.Module):
    """
    Frozen ProtBERT feature extractor.

    Input:
        peptide sequences

    Output:
        mean pooled embedding
        [batch, hidden_size]
    """

    def __init__(
        self,
        model_path: str,
        max_len: int = 128,
        device=None,
    ):

        super().__init__()


        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


        self.max_len = max_len


        self.tokenizer = BertTokenizer.from_pretrained(
            model_path,
            do_lower_case=False
        )


        self.model = BertModel.from_pretrained(
            model_path
        )


        self.hidden_size = (
            self.model.config.hidden_size
        )


        # freeze ProtBERT

        for p in self.model.parameters():

            p.requires_grad = False


        self.model.eval()


        self.model.to(self.device)


    @torch.no_grad()
    def forward(
        self,
        sequences: List[str]
    ):


        # ProtBERT requires:
        # A C D E F


        spaced_sequences = [
            " ".join(list(seq))
            for seq in sequences
        ]


        inputs = self.tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )


        input_ids = (
            inputs["input_ids"]
            .to(self.device)
        )


        attention_mask = (
            inputs["attention_mask"]
            .to(self.device)
        )


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        hidden_states = (
            outputs.last_hidden_state
        )


        # -----------------------------
        # mean pooling
        # remove CLS SEP PAD
        # -----------------------------


        mask = attention_mask.clone()


        special_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ]


        for token_id in special_tokens:

            if token_id is not None:

                mask[
                    input_ids == token_id
                ] = 0


        mask_expand = (
            mask.unsqueeze(-1)
        )


        summed = (
            hidden_states * mask_expand
        ).sum(dim=1)


        lengths = (
            mask.sum(dim=1)
            .unsqueeze(-1)
            .clamp(min=1)
        )


        embedding = (
            summed / lengths
        )


        return embedding.cpu()


# ==========================================================
# Dataset
# ==========================================================


class FeatureDataset(Dataset):

    def __init__(
        self,
        features,
        labels,
    ):

        self.features = features

        self.labels = labels


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, idx):

        return (
            self.features[idx],
            self.labels[idx]
        )


# ==========================================================
# MLP prediction head
# ==========================================================


class MLPHead(nn.Module):

    def __init__(
        self,
        input_dim,
        task,
        hidden_dims=(256,128),
        dropout=0.1,
    ):

        super().__init__()


        layers = []


        last_dim = input_dim


        for dim in hidden_dims:


            layers.append(
                nn.Linear(
                    last_dim,
                    dim
                )
            )


            layers.append(
                nn.ReLU()
            )


            layers.append(
                nn.Dropout(dropout)
            )


            last_dim = dim


        if task == "classification":

            layers.append(
                nn.Linear(
                    last_dim,
                    1
                )
            )


            layers.append(
                nn.Sigmoid()
            )


        else:


            layers.append(
                nn.Linear(
                    last_dim,
                    1
                )
            )


        self.network = nn.Sequential(
            *layers
        )


    def forward(self,x):

        return self.network(x)


# ==========================================================
# Metric calculation
# ==========================================================


def calculate_metrics(
    y_true,
    y_pred,
    task
):


    result = {}


    if task == "classification":


        pred_label = (
            y_pred >= 0.5
        ).astype(int)


        result["accuracy"] = (
            accuracy_score(
                y_true,
                pred_label
            )
        )


        result["precision"] = (
            precision_score(
                y_true,
                pred_label,
                zero_division=0
            )
        )


        result["recall"] = (
            recall_score(
                y_true,
                pred_label,
                zero_division=0
            )
        )


        result["f1"] = (
            f1_score(
                y_true,
                pred_label,
                zero_division=0
            )
        )


        try:

            result["roc_auc"] = (
                roc_auc_score(
                    y_true,
                    y_pred
                )
            )

        except:

            result["roc_auc"] = None


        try:

            result["auprc"] = (
                average_precision_score(
                    y_true,
                    y_pred
                )
            )

        except:

            result["auprc"] = None


    else:


        result["mse"] = (
            mean_squared_error(
                y_true,
                y_pred
            )
        )


        result["rmse"] = math.sqrt(
            result["mse"]
        )


        result["mae"] = (
            mean_absolute_error(
                y_true,
                y_pred
            )
        )


        result["r2"] = (
            r2_score(
                y_true,
                y_pred
            )
        )


    return result
# ==========================================================
# Data loading
# ==========================================================


def infer_task_type(
    labels
):

    unique_values = np.unique(labels)


    # binary label

    if len(unique_values) == 2:

        if set(unique_values).issubset(
            {0,1,0.0,1.0}
        ):

            return "classification"


    return "regression"


def collect_csv_files(
    data_dir,
    datasets=None
):

    data_dir = Path(data_dir)


    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )


    files = list(
        data_dir.glob("*.csv")
    )


    if datasets is not None and len(datasets)>0:

        files = [
            f for f in files
            if f.stem in datasets
        ]


    return files


def load_dataset(
    csv_file
):

    df = pd.read_csv(csv_file)


    # --------------------------
    # sequence column detection
    # --------------------------

    seq_candidates = [
        "peps",
        "sequence",
        "seq",
        "Sequence",
        "SEQ"
    ]


    label_candidates = [
        "label",
        "Label",
        "target",
        "value",
        "activity"
    ]


    seq_col = None

    label_col = None


    for c in seq_candidates:

        if c in df.columns:

            seq_col=c
            break


    for c in label_candidates:

        if c in df.columns:

            label_col=c
            break


    if seq_col is None:

        raise ValueError(
            f"No sequence column in {csv_file}"
        )


    if label_col is None:

        raise ValueError(
            f"No label column in {csv_file}"
        )


    sequences = (
        df[seq_col]
        .astype(str)
        .tolist()
    )


    labels = (
        df[label_col]
        .values
    )


    task = infer_task_type(
        labels
    )


    if task=="classification":

        labels = labels.astype(
            np.float32
        )

    else:

        labels = labels.astype(
            np.float32
        )


    return (
        sequences,
        labels,
        task
    )


# ==========================================================
# Data split
# ==========================================================


def build_split_dataframe(
    sequences: List[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build the splitter-compatible table and retain feature row identities."""
    data = pd.DataFrame({"peps": sequences, "label": labels})
    data["_sample_id"] = np.arange(len(data), dtype=np.int64)
    return data


def split_dataset_for_seed(
    data: pd.DataFrame,
    task_type: str,
    split_method: str,
    test_size: float,
    val_size: float,
    seed: int,
    similarity_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match the ESM search split contract for one reproducible seed."""
    if split_method == "similarity":
        train_data, test_data, val_data = split_dataset_by_similarity(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            similarity_threshold=similarity_threshold,
        )
    elif split_method == "random":
        train_data, test_data, val_data = split_dataset(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            stratify=(task_type == "classification"),
        )
    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    for split_name, frame in (
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ):
        if frame is None or len(frame) == 0:
            raise ValueError(f"{split_name} split is empty")
        if "_sample_id" not in frame.columns:
            raise RuntimeError(
                "The splitter did not preserve '_sample_id'; frozen features "
                "cannot be indexed safely."
            )

    return (
        train_data.reset_index(drop=True).copy(),
        test_data.reset_index(drop=True).copy(),
        val_data.reset_index(drop=True).copy(),
    )


def prepare_split_contexts(
    data: pd.DataFrame,
    task_type: str,
    split_method: str,
    seeds: List[int],
    test_size: float,
    val_size: float,
    similarity_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    """Prepare fixed split indices before learning-rate search, as in ESM."""
    contexts: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        train_data, test_data, val_data = split_dataset_for_seed(
            data=data,
            task_type=task_type,
            split_method=split_method,
            test_size=test_size,
            val_size=val_size,
            seed=int(seed),
            similarity_threshold=similarity_threshold,
        )
        contexts[int(seed)] = {
            "train_ids": train_data["_sample_id"].to_numpy(dtype=np.int64),
            "validation_ids": val_data["_sample_id"].to_numpy(dtype=np.int64),
            "test_ids": test_data["_sample_id"].to_numpy(dtype=np.int64),
            "split_sizes": {
                "train": int(len(train_data)),
                "validation": int(len(val_data)),
                "test": int(len(test_data)),
            },
        }
        print(
            f"Prepared split seed={seed}: train={len(train_data)}, "
            f"validation={len(val_data)}, test={len(test_data)}"
        )
    return contexts


# ==========================================================
# Feature extraction
# ==========================================================


def extract_features(
    encoder,
    sequences,
    batch_size=16
):


    encoder.eval()


    all_features=[]


    for i in range(
        0,
        len(sequences),
        batch_size
    ):


        batch_sequences = (
            sequences[
                i:i+batch_size
            ]
        )


        with torch.no_grad():

            emb = encoder(
                batch_sequences
            )


        all_features.append(
            emb
        )


    features = torch.cat(
        all_features,
        dim=0
    )


    return features


# ==========================================================
# Training utilities
# ==========================================================


def create_loader(
    features,
    labels,
    indices,
    batch_size,
    shuffle
):


    x = features[
        indices
    ]


    y = torch.tensor(
        labels[
            indices
        ],
        dtype=torch.float32
    )


    dataset = FeatureDataset(
        x,
        y
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


    return loader


def evaluate_model(
    model,
    loader,
    device,
    task
):


    model.eval()


    all_pred=[]
    all_true=[]


    with torch.no_grad():

        for x,y in loader:


            x=x.to(device)

            y=y.to(device)


            output=model(x)


            output=(
                output
                .view(-1)
                .cpu()
                .numpy()
            )


            y=(
                y
                .view(-1)
                .cpu()
                .numpy()
            )


            all_pred.extend(
                output.tolist()
            )


            all_true.extend(
                y.tolist()
            )


    metrics = calculate_metrics(
        np.array(all_true),
        np.array(all_pred),
        task
    )


    return metrics


def train_one_model(
    model,
    train_loader,
    val_loader,
    task,
    device,
    learning_rate,
    max_epochs,
    patience,
    checkpoint_path=None,
):


    model.to(device)


    if task=="classification":

        criterion = nn.BCELoss()

        monitor = "f1"

        higher_better=True


    else:

        criterion = nn.MSELoss()

        monitor="rmse"

        higher_better=False


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )


    best_metric = (
        -1e9
        if higher_better
        else 1e9
    )


    best_epoch=0


    best_state=None


    patience_count=0


    start_epoch=0


    # -----------------------
    # resume
    # -----------------------

    if checkpoint_path and os.path.exists(
        checkpoint_path
    ):


        ckpt=torch_load_compatible(
            checkpoint_path,
            map_location=device
        )


        model.load_state_dict(
            ckpt["model"]
        )


        optimizer.load_state_dict(
            ckpt["optimizer"]
        )


        start_epoch=(
            ckpt["epoch"]+1
        )


        best_metric=(
            ckpt["best_metric"]
        )


        best_epoch=(
            ckpt["best_epoch"]
        )


    for epoch in range(
        start_epoch,
        max_epochs
    ):


        model.train()


        for x,y in train_loader:


            x=x.to(device)

            y=y.to(device)


            optimizer.zero_grad()


            pred=model(x)


            loss=criterion(
                pred.view(-1),
                y.view(-1)
            )


            loss.backward()


            optimizer.step()


        val_metrics=evaluate_model(
            model,
            val_loader,
            device,
            task
        )


        current=(
            val_metrics[monitor]
        )


        improved = (
            current > best_metric
            if higher_better
            else current < best_metric
        )


        if improved:


            best_metric=current

            best_epoch=epoch

            best_state={
                k:v.cpu().clone()
                for k,v in model.state_dict().items()
            }

            patience_count=0


        else:

            patience_count+=1


        if checkpoint_path:


            torch.save(
                {
                    "epoch":epoch,
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "best_metric":best_metric,
                    "best_epoch":best_epoch
                },
                checkpoint_path
            )


        if patience_count >= patience:

            break


    if best_state:

        model.load_state_dict(
            best_state
        )


    return (
        model,
        best_epoch,
        best_metric
    )

# ==========================================================
# Result saving utilities
# ==========================================================


def append_json(
    file_path,
    record
):

    file_path = Path(file_path)


    if file_path.exists():

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

    else:

        data=[]


    data.append(record)


    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def append_csv(
    file_path,
    record
):

    file_path=Path(file_path)


    exists=file_path.exists()


    with open(
        file_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer=csv.DictWriter(
            f,
            fieldnames=list(record.keys())
        )


        if not exists:

            writer.writeheader()


        writer.writerow(record)


# ==========================================================
# Save best model
# ==========================================================


def save_model_checkpoint(
    model,
    save_path,
    metadata
):


    save_path=Path(save_path)


    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "metadata":
                metadata
        },
        save_path
    )


# ==========================================================
# One dataset experiment
# ==========================================================


def train_dataset(
    dataset_file,
    args,
    encoder,
    device
):


    dataset_name=(
        Path(dataset_file)
        .stem
    )


    print(
        f"\n========== {dataset_name} =========="
    )


    sequences, labels, task = load_dataset(
        dataset_file
    )


    print(
        "Task:",
        task,
        "Samples:",
        len(labels)
    )


    # ---------------------------------
    # Extract ProtBERT features once
    # ---------------------------------

    print(
        "Extracting ProtBERT features..."
    )


    features = extract_features(
        encoder,
        sequences,
        batch_size=args.feature_batch_size
    )


    input_dim=features.shape[1]


    print(
        "Embedding dimension:",
        input_dim
    )


    split_data = build_split_dataframe(sequences, labels)

    for split_method in args.split_methods:

        contexts = prepare_split_contexts(
            data=split_data,
            task_type=task,
            split_method=split_method,
            seeds=args.seeds,
            test_size=args.test_size,
            val_size=args.val_size,
            similarity_threshold=args.similarity_threshold,
        )

        for seed in args.seeds:


            print(
                f"\nSplit={split_method}, Seed={seed}"
            )


            set_seed(seed)


            context = contexts[int(seed)]
            train_idx = context["train_ids"]
            val_idx = context["validation_ids"]
            test_idx = context["test_ids"]


            # =============================
            # Train with the fixed learning rate
            # =============================


            lr_results=[]


            for lr in args.learning_rates:


                key_name=f"{dataset_name}_{split_method}_{seed}_{lr}"


                checkpoint_dir=Path(
                    args.resume_dir
                )

                checkpoint_dir.mkdir(
                    exist_ok=True,
                    parents=True
                )


                checkpoint_path=(
                    checkpoint_dir /
                    f"{key_name}.pt"
                )


                print(
                    "Learning rate:",
                    lr
                )


                train_loader=create_loader(
                    features,
                    labels,
                    train_idx,
                    args.batch_size,
                    True
                )


                val_loader=create_loader(
                    features,
                    labels,
                    val_idx,
                    args.batch_size,
                    False
                )


                model=MLPHead(
                    input_dim=input_dim,
                    task=task,
                    hidden_dims=(256,128),
                    dropout=0.1
                )


                model,best_epoch,best_metric = (
                    train_one_model(
                        model,
                        train_loader,
                        val_loader,
                        task,
                        device,
                        lr,
                        args.max_epochs,
                        args.patience,
                        checkpoint_path
                    )
                )


                val_metrics=evaluate_model(
                    model,
                    val_loader,
                    device,
                    task
                )


                record={

                    "dataset":
                        dataset_name,

                    "split_method":
                        split_method,

                    "similarity_threshold":
                        args.similarity_threshold
                        if split_method == "similarity"
                        else None,

                    "split_sizes":
                        context["split_sizes"],

                    "seed":
                        seed,

                    "model":
                        "ProtBERT",

                    "learning_rate":
                        lr,

                    "task":
                        task,

                    "best_epoch":
                        best_epoch,

                    **val_metrics
                }


                append_json(
                    args.output_json,
                    record
                )


                append_csv(
                    args.search_csv,
                    record
                )


                lr_results.append(
                    record
                )


            # =============================
            # Use the fixed learning rate
            # =============================


            if task=="classification":


                best_record=max(
                    lr_results,
                    key=lambda x:x["f1"]
                )


            else:


                best_record=min(
                    lr_results,
                    key=lambda x:x["rmse"]
                )


            best_lr=(
                best_record["learning_rate"]
            )


            best_epoch=(
                best_record["best_epoch"]
            )


            print(
                "Learning rate:",
                best_lr,
                "Epoch:",
                best_epoch
            )


            # =============================
            # Retrain with train+val
            # =============================


            final_train_idx=np.concatenate(
                [
                    train_idx,
                    val_idx
                ]
            )


            final_loader=create_loader(
                features,
                labels,
                final_train_idx,
                args.batch_size,
                True
            )


            test_loader=create_loader(
                features,
                labels,
                test_idx,
                args.batch_size,
                False
            )


            final_model=MLPHead(
                input_dim=input_dim,
                task=task,
                hidden_dims=(256,128),
                dropout=0.1
            )


            final_model.to(device)


            if task=="classification":

                criterion=nn.BCELoss()

            else:

                criterion=nn.MSELoss()


            optimizer=torch.optim.Adam(
                final_model.parameters(),
                lr=best_lr,
                weight_decay=1e-5
            )


            for epoch in range(
                best_epoch+1
            ):


                final_model.train()


                for x,y in final_loader:


                    x=x.to(device)

                    y=y.to(device)


                    optimizer.zero_grad()


                    pred=final_model(x)


                    loss=criterion(
                        pred.view(-1),
                        y.view(-1)
                    )


                    loss.backward()


                    optimizer.step()


            test_metrics=evaluate_model(
                final_model,
                test_loader,
                device,
                task
            )


            # =============================
            # Save model
            # =============================


            model_path=(
                Path(args.best_model_dir)
                /
                split_method
                /
                dataset_name
                /
                f"protbert_mlp_seed{seed}.pt"
            )


            metadata={

                "model":
                    "protbert_mlp",

                "feature_type":
                    "protbert",

                "task_type":
                    task,

                "freeze_encoder":
                    True,

                "pooling":
                    "mean",

                "embedding_dim":
                    int(input_dim),

                "dataset":
                    dataset_name,

                "split_method":
                    split_method,

                "similarity_threshold":
                    args.similarity_threshold
                    if split_method == "similarity"
                    else None,

                "split_sizes":
                    context["split_sizes"],

                "seed":
                    seed,

                "learning_rate":
                    best_lr,

                "best_parameters": {
                    "learning_rate": best_lr,
                    "hidden_dims": [256, 128],
                    "dropout": 0.1,
                    "weight_decay": 1e-5,
                },

                "selection_metric":
                    "f1" if task == "classification" else "rmse",

                "selection_direction":
                    "maximize" if task == "classification" else "minimize",

                "validation_summary": {
                    name: {"mean": value, "std": 0.0, "n": 1}
                    for name, value in val_metrics.items()
                    if value is not None
                },

                "best_epoch":
                    best_epoch,

                "test_metrics":
                    test_metrics

            }


            save_model_checkpoint(
                final_model,
                model_path,
                metadata
            )


            metric_path=(
                Path(args.best_metrics_dir)
                /
                split_method
                /
                dataset_name
                /
                f"seed{seed}.json"
            )


            write_metrics_report(metric_path, metadata)


# ==========================================================
# Main experiment
# ==========================================================


def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--data_dir",
        type=str,
        default="pephub/raw_data"
    )


    parser.add_argument(
        "--protbert_model_path",
        type=str,
        default="pretrained/prot_bert"
    )


    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None
    )


    parser.add_argument(
        "--split_methods",
        nargs="+",
        default=[
            "random",
            "similarity"
        ]
    )


    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42,43,44]
    )


    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE
    )


    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )


    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1
    )


    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8
    )


    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=16
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )


    parser.add_argument(
        "--max_len",
        type=int,
        default=128
    )


    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=10
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )


    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/protbert/run_state.json"
    )


    parser.add_argument(
        "--search_csv",
        type=str,
        default="outputs/protbert/training_runs.csv"
    )


    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="outputs/models/protbert"
    )


    parser.add_argument(
        "--best_metrics_dir",
        type=str,
        default="outputs/metrics/protbert"
    )


    parser.add_argument(
        "--resume_dir",
        type=str,
        default="outputs/checkpoints/protbert"
    )


    args=parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    args.learning_rates = [args.learning_rate]


    device=get_device(
        args.device
    )


    print(
        "Using device:",
        device
    )


    encoder=ProtBERTEncoder(
        args.protbert_model_path,
        args.max_len,
        device
    )


    csv_files=collect_csv_files(
        args.data_dir,
        args.datasets
    )


    for csv_file in csv_files:

        train_dataset(
            csv_file,
            args,
            encoder,
            device
        )


if __name__=="__main__":

    main()

