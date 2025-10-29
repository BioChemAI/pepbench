# ===== train.py =====
import os
import joblib
import argparse
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from dataset import PepDataset
from model_manager import ModelManager
from model.predict_model import PredictModel


# wandb
try:
    import wandb
except ImportError:
    wandb = None

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train peptide models.")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                        help="Specify task type: 'classification' or 'regression'.")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name, e.g., rf | svm | xgb | lstm | transformer | esm | pepbert.")
    parser.add_argument('--data_name', type=str, required=True,
                        help="Dataset identifier for naming.")
    parser.add_argument('--train_path', type=str, required=True,
                        help="Path to training CSV file.")
    parser.add_argument("--val_path", type=str, required=True, 
                        help="Path to validation CSV file.")
    parser.add_argument('--feature_type', type=str, default='',
                        help="Feature encoding type (e.g., onehot, descriptor, etc.).")
    parser.add_argument('--max_len', type=int, default=20,
                        help="Maximum peptide sequence length.")
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='saved_models')
    # parser.add_argument('--train_embed_path', type=str, default=None)
    # parser.add_argument('--val_embed_path', type=str, default=None)
    parser.add_argument('--freeze', action='store_true', help='Freeze pre-trained layers and only train classifier')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Task: {args.task} | Model: {args.model}")

    # === 0. Initialize wandb ===
    use_wandb = (not args.no_wandb) and (wandb is not None)
    if use_wandb:
        wandb.init(
            project="pepbench-train",
            name=f"{args.model}-{args.feature_type}-{args.data_name}-{args.task}",
            config={
                "task": args.task,
                "model": args.model,
                "data_name": args.data_name,
                "train_path": args.train_path,
                "val_path": args.val_path,
                "feature_type":args.feature_type,
                "max_len": args.max_len,
                "random_state": args.random_state,
                "output_dir": args.output_dir,
                "freeze": args.freeze
            }
        )

    # === 1. Data ===
    # 1.1 Load train data

    if args.model in ['rf', 'svm', 'xgb']:
        model_type = 'ml'
    elif args.model in ['lstm', 'transformer']:
        model_type = 'dl'
    else:
        model_type = 'll'

    train_dataset = PepDataset(
            csv_path=args.train_path,
            max_len=args.max_len,
            task=args.task,
            feature_type=args.feature_type,
            model_type=model_type
    )
    # 1.1 Load val data
    val_dataset = PepDataset(
            csv_path=args.val_path,
            max_len=args.max_len,
            task=args.task,
            feature_type=args.feature_type,
            model_type=model_type
    )

    if args.model in ['transformer', 'lstm', 'esm', 'pepbert']:
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    else:
        features_train = train_dataset.features
        features_val = val_dataset.features

    y_train = train_dataset.labels
    y_val = val_dataset.labels

    # === 2. Model manager ===
    manager = ModelManager()
    # 2.1 Load model
    model = manager.create_model(
        name=args.model,
        task=args.task,
        max_len=args.max_len,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        random_state=args.random_state
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'll':
        if args.model == 'esm':
            predicts_model = PredictModel(hidden_size=480, task=args.task)   # esm:480
        else:
            predicts_model = PredictModel(hidden_size=1024, task=args.task)  # prot_bert:1024
        
        predicts_model = predicts_model.to(device)
    if model_type in ['dl', 'll']:
        model = model.to(device)

    # 2.2 Freeze pretrained parameters
    if args.freeze and model_type=='ll':
        for name, param in model.named_parameters():
            param.requires_grad = False
        print("[INFO] Pre-trained backbone frozen, only classifier will be trained.")

    # === 3. Train ===
    # 3.1 Initialize the best indicators
    best_val_accuracy = 0.0 if args.task == 'classification' else float('inf')
    best_val_loss = float('inf')
    best_epoch = 0

    # 3.2 Best model saving path
    best_model_path = manager.get_best_model_path(args.output_dir, args.model, args.feature_type, args.task, args.data_name, args.random_state)
    # 3.3 Training
    if model_type == 'ml':
        print("Training traditional machine learning model...")
        model.fit(features_train, y_train)
        if args.task == 'classification':
            # Classification task: Predict probability
            val_predictions = model.predict(features_val)
            train_predictions = model.predict(features_train)

            # Calculate accuracy rate
            val_accuracy = accuracy_score(y_val, val_predictions > 0.5)
            train_accuracy = accuracy_score(y_train, train_predictions > 0.5)

            # Calculate the loss
            train_loss = log_loss(y_train, train_predictions)
            val_loss = log_loss(y_val, val_predictions)

        else:
            # Regression task: Direct prediction
            val_predictions = model.predict(features_val)
            train_predictions = model.predict(features_train)

            # Calculate the MSE loss
            train_loss = mean_squared_error(y_train, train_predictions)
            val_loss = mean_squared_error(y_val, val_predictions)

        # Logged to wandb
        if use_wandb:
            log_data = {
                "epoch": 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            if args.task == 'classification':
                log_data.update({
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                })
            wandb.log(log_data)
        # Print result
        if args.task == 'classification':
            print(f"[Final] Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        else:
            print(f"[Final] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model (traditional models only have one training session)
        joblib.dump(model, best_model_path)
        print(f"✅ Best model saved to: {best_model_path}")
    elif model_type == 'dl':
        print("Training deep learning model...")
        patience = 100
        no_improve_count = 0
        criterion = nn.BCELoss() if args.task == 'classification' else nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)

        for epoch in range(300):
            model.train()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in train_loader:
                optimizer.zero_grad()
                seqs, labels = batch
                labels = labels.to(device)
                outputs = model(seqs)
                outputs = outputs.squeeze()

                if args.task == 'classification':
                    labels = labels.float()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate the training accuracy
                if args.task == 'classification':
                    with torch.no_grad():
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        total_correct += correct
                        total_samples += labels.size(0)

            # Calculate the average indicator
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = total_correct / total_samples if args.task == 'classification' else None

            # Learning rate scheduling
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # ===== Verification =====
            model.eval()

            val_loss = 0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    seqs, labels = batch
                    outputs = model(seqs)
                    labels = labels.to(device)
                    outputs = outputs.squeeze()

                    if args.task == 'classification':
                        labels = labels.float()

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate the verification accuracy rate
                    if args.task == 'classification':
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        val_correct += correct
                        val_samples += labels.size(0)

            # Calculate the average validation metric
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_samples if args.task == 'classification' else None

            # ===== Logged to wandb =====
            if use_wandb:
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr
                }

                if args.task == 'classification':
                    log_data.update({
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy
                    })

                wandb.log(log_data)

            # ===== Print information =====
            if args.task == 'classification':
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # ===== Save the best model =====
            if args.task == 'classification':
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                    no_improve_count = 0
                    print(f"✅ New best model saved with val accuracy: {val_accuracy:.4f}")
                else:
                    no_improve_count += 1
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                    no_improve_count = 0
                    print(f"✅ New best model saved with val loss: {avg_val_loss:.4f}")
                else:
                    no_improve_count += 1

            # ===== Early stopping check =====
            if no_improve_count >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

        print(f"\n Training completed! Best model from epoch {best_epoch}")
        if args.task == 'classification':
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"Best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print("Training predict_model...")
        patience = 50
        no_improve_count = 0
        criterion = nn.BCELoss() if args.task == 'classification' else nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, predicts_model.parameters()), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)

        for epoch in range(200):
            predicts_model.train()
            model.eval()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in train_loader:
                optimizer.zero_grad()
                seqs, labels = batch
                labels = labels.to(device)

                embed = model(sequences=seqs)
                outputs = predicts_model(embed)
                outputs = outputs.squeeze()


                if args.task == 'classification':
                    labels = labels.float()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate the training accuracy
                if args.task == 'classification':
                    with torch.no_grad():
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        total_correct += correct
                        total_samples += labels.size(0)

            # Calculate the average indicator
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = total_correct / total_samples if args.task == 'classification' else None

            # Learning rate scheduling
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # ===== Verification =====
            predicts_model.eval()
            val_loss = 0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    seqs, labels = batch
                    embed = model(sequences=seqs)
                    # embed = seqs.to(device) #pepbert_embed
                    outputs = predicts_model(embed)
                    labels = labels.to(device)
                    outputs = outputs.squeeze()

                    if args.task == 'classification':
                        labels = labels.float()

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate the verification accuracy rate
                    if args.task == 'classification':
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        val_correct += correct
                        val_samples += labels.size(0)

            # Calculate the average validation metric
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_samples if args.task == 'classification' else None

            # ===== Logged to wandb =====
            if use_wandb:
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr
                }

                if args.task == 'classification':
                    log_data.update({
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy
                    })

                wandb.log(log_data)

            # ===== Print information =====
            if args.task == 'classification':
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # ===== Save the best model =====
            if args.task == 'classification':
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(predicts_model.state_dict(), best_model_path) # esm/pepbert
                    no_improve_count = 0
                    print(f"✅ New best model saved with val accuracy: {val_accuracy:.4f}")
                else:
                    no_improve_count += 1
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(predicts_model.state_dict(), best_model_path) # esm/pepbert
                    no_improve_count = 0
                    print(f"✅ New best model saved with val loss: {avg_val_loss:.4f}")
                else:
                    no_improve_count += 1


    # === 4. Close wandb ===
    if use_wandb:
        wandb.finish()
if __name__ == '__main__':
    main()
