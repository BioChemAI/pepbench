# ===== train.py =====
import os
import time
import joblib
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from feature.OneHot import OneHotEncoder
from model.factory import build_model

# 支持的深度模型名称（用于是否使用 flatten）
TRANSFORMER_MODELS = ['transformer', 'lstm', 'esm', 'pepbert']


def parse_args():
    parser = argparse.ArgumentParser(description="Train peptide models.")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                        help="Task type: classification or regression")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name: rf, svm, logistic, xgb, transformer, etc.")
    parser.add_argument('--max_len', type=int, default=20,
                        help="Max sequence length for encoding")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--output_dir', type=str, default='saved_models',
                        help="Directory to save trained models")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to the training dataset CSV file")
    parser.add_argument("--data_name", type=str, required=True,
                        help="name of the training dataset CSV file")
    parser.add_argument('--input_dim', type=int, default=20,
                    help="Input dimension for LSTM or transformer (e.g., 20 for one-hot, 128 for embeddings)")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Task: {args.task} | Model: {args.model}")
    train_path = args.train_path
    data_name = args.data_name
    # ==== Step 2: 加载训练集 ====
    df = pd.read_csv(train_path)
    X_train = df['peps'].values
    y_train = df['label'].values.astype(np.float32 if args.task == 'regression' else int)

    # ==== Step 3: 特征编码 ====
    if args.model in ['esm', 'pepbert']:
        features_train = X_train  # 直接使用序列输入
    else:
        encoder = OneHotEncoder(max_len=args.max_len, flatten=True)
        features_train = np.array([encoder.encode(seq) for seq in X_train])
        if features_train.ndim == 3 and args.model not in ['lstm', 'transformer']:
            features_train = features_train.reshape(features_train.shape[0], -1)

    # ==== Step 4: 设置设备 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ==== Step 5: 构建模型 ====
    if args.model in TRANSFORMER_MODELS:
        model = build_model(name=args.model, 
                            task=args.task, 
                            device=device, 
                            max_len=args.max_len,
                            input_dim=args.input_dim) # lstm
                            #model_path='MODEL/prot_bert') # esm,pepbert

        # 构建训练数据加载器
        from torch.utils.data import Dataset, DataLoader

        class SequenceDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences = sequences
                self.labels = torch.tensor(labels, dtype=torch.float32 if args.task == 'regression' else torch.long)

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                x = self.sequences[idx]
                if isinstance(x, np.ndarray) and x.ndim == 1:
                    # 恢复成 [L, D]
                    x = x.reshape(args.max_len, args.input_dim)
                return torch.tensor(x, dtype=torch.float32), self.labels[idx]
            
            #esm,pepbert用：
            # def __getitem__(self, idx):
            #     x = self.sequences[idx]

            #     if args.model not in ["esm", "pepbert"]:
            #         if isinstance(x, np.ndarray) and x.ndim == 1:
            #             x = x.reshape(args.max_len, args.input_dim)
            #         x = torch.tensor(x, dtype=torch.float32)

            #     return x, self.labels[idx]




        train_dataset = SequenceDataset(features_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # 设置损失函数与优化器
        if args.task == 'classification':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.to(device)
        model.train()

        # ==== Step 6: 模型训练 ====
        for epoch in range(50):
            total_loss = 0
            for seqs, labels in train_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                if args.task == 'classification':
                    labels = labels.float().unsqueeze(1).to(device)# lstm,rf
                    #labels = labels.float().to(device)# esm

                optimizer.zero_grad()
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    else:
        # 传统模型训练流程
        input_dim = features_train.shape[1]
        model = build_model(name=args.model, task=args.task, input_dim=input_dim)

        model.train(features_train, y_train)

    # ==== Step 7: 保存模型 ====
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "pt" if args.model in TRANSFORMER_MODELS else "pkl"
    model_path = os.path.join(args.output_dir,
                              f"{args.model}_{args.task}_{args.data_name}_seed{args.random_state}_{timestamp}.{suffix}")

    if args.model in TRANSFORMER_MODELS:
        torch.save(model.state_dict(), model_path)
    else:
        joblib.dump(model, model_path)

    print(f"\n[INFO] Model saved to: {model_path}")


if __name__ == '__main__':
    main()
