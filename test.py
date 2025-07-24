# ===== test.py =====
import os
import joblib
import numpy as np
import pandas as pd
import torch
import argparse

from feature.OneHot import OneHotEncoder
from model.factory import build_model
from utils.metrics import evaluate_classification, evaluate_regression

# 支持的深度模型名称（用于是否使用序列原始输入）
TRANSFORMER_MODELS = ['transformer', 'lstm', 'esm']

# ==== Step 1: 加载测试集路径 ====
def parse_args():
    parser = argparse.ArgumentParser(description="Model testing arguments")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True,
                        help="Task type: classification or regression")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name: rf, svm, transformer, esm, pepbert, etc.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model file")
    parser.add_argument('--test_path', type=str, required=True,
                        help="Path to the test dataset CSV file")
    parser.add_argument('--max_len', type=int, default=190,
                        help="Maximum sequence length (used for transformer-based models)")
    return parser.parse_args()

args = parse_args()
test_path = args.test_path
task = args.task
model_name = args.model
model_path = args.model_path
max_len = args.max_len

print(f"[INFO] Task: {task} | Model: {model_name}")
print(f"[INFO] Loading model from: {model_path}")

# ==== Step 2: 加载测试数据 ====
df = pd.read_csv(test_path)
X_test = df['peps'].values
y_test = df['label'].values.astype(np.float32 if task == 'regression' else int)

# ==== Step 3: 特征编码 ====
if model_name == 'esm':
    features_test = X_test  # 直接使用序列
else:
    encoder = OneHotEncoder(max_len=max_len, flatten=True)
    features_test = np.array([encoder.encode(seq) for seq in X_test])
    if features_test.ndim == 3:
        features_test = features_test.reshape(features_test.shape[0], -1)

# ==== Step 4: 设置设备 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==== Step 5: 加载模型并预测 ====
if model_name in TRANSFORMER_MODELS:
    model = build_model(name=model_name, task=task, device=device, max_len=max_len)
    #model = build_model(name=model_name, task=task, device=device, max_len=max_len, input_dim = 20)# lstm
    #model = build_model(name=model_name, task=task, device=device, max_len=max_len, model_path='MODEL/esm2_t12_35M_UR50D')# eam,pepbert
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 构建测试数据加载器
    from torch.utils.data import Dataset, DataLoader

    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = torch.tensor(labels, dtype=torch.float32 if task == 'regression' else torch.long)

        def __len__(self):
            return len(self.sequences)
        # lstm
        def __getitem__(self, idx):
            x = self.sequences[idx]
            if isinstance(x, np.ndarray) and x.ndim == 1:
                x = x.reshape(max_len, 20)
            return torch.tensor(x, dtype=torch.float32), self.labels[idx]
        #esm用：
        # def __getitem__(self, idx):
        #     return self.sequences[idx], self.labels[idx]


    test_dataset = SequenceDataset(features_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 执行预测
    all_preds = []
    with torch.no_grad():
        for seqs, _ in test_loader:
            outputs = model(seqs)  # shape (B, 1)
            probs = torch.sigmoid(outputs)  # 转成概率
            preds = (probs > 0.5).long()    # 转成0/1
            all_preds.extend(preds.cpu().numpy())
    y_pred = np.array(all_preds).flatten()


else:
    # 传统模型预测流程
    model = joblib.load(model_path)
    y_pred = model.predict(features_test)

# ==== Step 6: 评估结果 ====
print("\n[INFO] Evaluation on Test Set:")
if task == 'classification':
    metrics = evaluate_classification(y_test, y_pred)
else:
    metrics = evaluate_regression(y_test, y_pred)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")