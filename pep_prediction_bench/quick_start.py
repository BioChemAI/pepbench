import numpy as np

import torch
import joblib
from torch.utils.data import DataLoader

from dataset import PepDataset
from model_manager import ModelManager, model_type_identify
from model.esm_model import ESMModel
from model.pepbert import PepBERTModel
from utils.metrics import evaluate_classification, evaluate_regression

def compute_metrics(
        task='classification',
        data_path='data/assem_test.csv',
        data_max_len=24,
        feature_type='onehot',
        model='rf',
        model_path='saved_models/BEST_rf_onehot_classification_assem_seed111.pkl',
        batch_size=2048
):
    # model type
    model_type = model_type_identify(model)
    
    # data load
    test_dataset = PepDataset(
            csv_path=data_path,
            max_len=data_max_len,
            task=task,
            feature_type=feature_type,
            model_type=model_type
    )

    if model_type in ['dl', 'll']:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        features_test = test_dataset.features

    y_test = test_dataset.labels

    # model load
    manager = ModelManager()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = manager.load_model(
        path=model_path,
        name=model,
        task=task,
        max_len=data_max_len,
        device=device
    )

    if model_type in ['dl', 'll']:
        model.to(device)
        model.eval()
    else:
        model = joblib.load(model_path)
    
    # predict
    y_pred = None
    y_pred_prob = None
    all_preds = []

    if model_type == 'll':
        if model == 'esm':
            backbone = ESMModel(model_path="MODEL/esm2_t12_35M_UR50D", max_len=data_max_len, device=device)
        else:
            backbone = PepBERTModel(model_path="MODEL/prot_bert", max_len=data_max_len, device=device)

        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                embed = backbone(seqs)
                outputs = model(embed)
                all_preds.extend(outputs.cpu().numpy())

        y_pred = np.array(all_preds).flatten()
        y_pred_prob = y_pred

    elif model_type == 'dl':
        with torch.no_grad():
            for batch in test_loader:
                seqs, labels = batch
                outputs = model(seqs.to(device).squeeze(-1))
                all_preds.extend(outputs.cpu().numpy())

        y_pred = np.array(all_preds)
        y_pred_prob = y_pred

    else:
        if hasattr(model, 'predict_proba') and task == 'classification':
            y_pred_prob = model.predict_proba(features_test)[:, 1]
        else:
            y_pred_prob = model.predict(features_test)
        y_pred = (y_pred_prob >= 0.5).astype(int) if task == 'classification' else y_pred_prob
    
    # evaluate
    print("\n[INFO] Evaluation on Test Set:")
    if task == 'classification':
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)
        metrics = evaluate_classification(y_test, y_pred_binary)
    else:
        metrics = evaluate_regression(y_test, y_pred_prob)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
