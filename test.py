# ===== test.py =====
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader

from dataset import PepDataset
from model_manager import ModelManager
from utils.metrics import evaluate_classification, evaluate_regression
from model.esm_model import ESMModel
from model.pepbert import PepBERTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Model testing arguments")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    # parser.add_argument('--test_embed_path', type=str, default=None)
    parser.add_argument('--max_len', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Task: {args.task} | Model: {args.model}")
    print(f"[INFO] Loading model from: {args.model_path}")
    print(f"[INFO] Using device: {device}")

    # === 1. 加载测试数据 ===
    df = pd.read_csv(args.test_path)
    X_test = df['peps'].values
    y_test = df['label'].values.astype(np.float32 if args.task == 'regression' else int)

    # === 2. 构建 Dataset ===
    # pepbert
    # test_dataset = PepDataset(
    #     sequences=args.test_embed_path,
    #     task='classification',
    #     model_name='pepbert'
    # )
    # rf/svm/xgb/lstm/transformer/esm
    test_dataset = PepDataset(
        sequences=X_test,
        labels=y_test,
        task=args.task,
        max_len=args.max_len,
        model_name=args.model
    )
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # === 3. 初始化模型 ===
    manager = ModelManager()
    if args.model in ['rf', 'svn', 'xgb']:
        input_dim = 20 # onehot，其他模型不使用
    else:
        input_dim = None

    model = manager.load_model(
            path=args.model_path,
            name=args.model,
            task=args.task,
            max_len=args.max_len,
            input_dim=input_dim,
            device=device
        )
    model = model.to(device)
    # === 4. 执行预测 ===
    all_preds = []

    if args.model in ['esm', 'pepbert']:
        if args.model == 'esm':
            backbone = ESMModel(model_path="MODEL/esm2_t12_35M_UR50D", max_len=args.max_len, device=device)
        else:
            backbone = PepBERTModel(model_path="MODEL/prot_bert", max_len=args.max_len, device=device)

        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                # embed, labels = batch
                # outputs = model(embed.to(device))
                # embed = backbone(seqs)
                outputs = model(seqs)

                if args.task == 'classification':
                    preds = (outputs > 0.5).long()
                else:
                    preds = outputs#TODO:回归问题先不管

                all_preds.extend(preds.cpu().numpy())

        y_pred = np.array(all_preds).flatten()

    elif args.model in ['lstm', 'transformer']:
        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                outputs = model(seqs.to(device))

                if args.task == 'classification':
                    preds = (outputs > 0.5).long()
                else:
                    preds = outputs

                all_preds.extend(preds.cpu().numpy())

        y_pred = np.array(all_preds).flatten()

    else:
        # 传统机器学习模型
        y_pred_prob = model.predict(test_dataset.features)

    # === 5. 评估 ===
    print("\n[INFO] Evaluation on Test Set:")
    if args.task == 'classification':
        y_pred = (y_pred_prob >= 0.5).astype(int)
        metrics = evaluate_classification(y_test, y_pred)
    else:
        metrics = evaluate_regression(y_test, y_pred_prob)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
