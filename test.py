# ===== test.py =====
import numpy as np
import pandas as pd
import torch
import argparse
import joblib
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
    parser.add_argument('--feature_type', type=str, default='')
    parser.add_argument('--model_path', type=str, required=True,help="Path to the saved model file")
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Task: {args.task} | Model: {args.model}")
    print(f"[INFO] Loading model from: {args.model_path}")
    print(f"[INFO] Using device: {device}")

    # === 1. Test data load ===
    if args.model in ['rf', 'svm', 'xgb']:
        model_type = 'ml'
    elif args.model in ['lstm', 'transformer']:
        model_type = 'dl'
    else:
        model_type = 'll'

    test_dataset = PepDataset(
            csv_path=args.test_path,
            max_len=args.max_len,
            task=args.task,
            feature_type=args.feature_type,
            model_type=model_type
    )


    if args.model in ['transformer', 'lstm', 'esm', 'pepbert']:
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    else:
        features_test = test_dataset.features

    y_test = test_dataset.labels

    # === 2. Create model and load the weights ===
    manager = ModelManager()

    model = manager.load_model(
        path=args.model_path,
        name=args.model,
        task=args.task,
        max_len=args.max_len,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        random_state=111
    )

    if args.model in ['transformer', 'lstm', 'esm', 'pepbert']:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        model = joblib.load(args.model_path)

    # === 3. Predict ===
    y_pred = None
    y_pred_prob = None
    all_preds = []

    if args.model in ['esm', 'pepbert']:
        if args.model == 'esm':
            backbone = ESMModel(model_path="MODEL/esm2_t12_35M_UR50D", max_len=args.max_len, device=device)
        else:
            backbone = PepBERTModel(model_path="MODEL/prot_bert", max_len=args.max_len, device=device)

        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                embed = backbone(seqs)
                outputs = model(embed)
                all_preds.extend(outputs.cpu().numpy())

        y_pred = np.array(all_preds).flatten()
        y_pred_prob = y_pred

    elif args.model in ['lstm', 'transformer']:
        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                outputs = model(seqs.to(device).squeeze(-1))
                all_preds.extend(outputs.cpu().numpy())

        y_pred = np.array(all_preds)
        y_pred_prob = y_pred

    else:
        if hasattr(model, 'predict_proba') and args.task == 'classification':
            y_pred_prob = model.predict_proba(features_test)[:, 1]  # Positive class probability
        else:
            y_pred_prob = model.predict(features_test)
        y_pred = (y_pred_prob >= 0.5).astype(int) if args.task == 'classification' else y_pred_prob

    # === 5. Evaluate ===
    print("\n[INFO] Evaluation on Test Set:")
    if args.task == 'classification':
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)
        metrics = evaluate_classification(y_test, y_pred_binary)
    else:
        metrics = evaluate_regression(y_test, y_pred_prob)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
