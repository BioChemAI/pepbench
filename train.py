# ===== train.py =====
import os
import time
import joblib
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import random
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from dataset import PepDataset
from model_manager import ModelManager
from model.predict_model import PredictModel
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

# 新增：wandb
try:
    import wandb
except ImportError:
    wandb = None

# 统一随机种子设置
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train peptide models.")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='new_saved_models')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument('--data_name', type=str, required=True)
    # parser.add_argument('--train_embed_path', type=str, default=None)
    # parser.add_argument('--val_embed_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help="Path to resume model (optional)")
    parser.add_argument('--freeze', action='store_true', help='Freeze pre-trained layers and only train classifier')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Task: {args.task} | Model: {args.model}")

    # === 0. 初始化 wandb ===
    use_wandb = (not args.no_wandb) and (wandb is not None)
    if use_wandb:
        wandb.init(
            project="pepbench-train",
            name=f"{args.model}-{args.data_name}-{args.task}",
            config={
                "task": args.task,
                "model": args.model,
                "data_name": args.data_name,
                "max_len": args.max_len,
                "random_state": args.random_state,
                "freeze": args.freeze,
                "resume": args.resume if args.resume else "None",
                "train_path": args.train_path,
                "val_path": args.val_path,
                "output_dir": args.output_dir
            }
        )

    # === 1. 加载数据 ===
    # 加载训练数据集
    df_train = pd.read_csv(args.train_path)
    X_train = df_train['peps'].values
    y_train = df_train['label'].values.astype(np.float32 if args.task == 'regression' else int)
    # 用来验证
    # if len(X_train) > 100:
    #     X_train = X_train[:100]
    #     y_train = y_train[:100]
    train_pos = (y_train == 1).sum()
    train_neg = (y_train == 0).sum()
    print(f"[DATA INFO] 训练集: 正样本={train_pos}, 负样本={train_neg}, 总计={len(y_train)}")
    print(f"[DATA NAME]={args.data_name}")

    # pepbert
    # train_dataset = PepDataset(
    #     sequences=args.train_embed_path,
    #     task='classification',
    #     model_name='pepbert'
    # )
    # rf/svm/xgb/lstm/transformer/esm
    train_dataset = PepDataset(
            sequences=X_train,
            labels=y_train,
            task=args.task,
            max_len=args.max_len,
            model_name=args.model
    )

    # 加载验证数据集
    df_val = pd.read_csv(args.val_path)
    X_val = df_val['peps'].values
    y_val = df_val['label'].values.astype(np.float32 if args.task == 'regression' else int)

    # pepbert
    # val_dataset = PepDataset(
    #     sequences=args.val_embed_path,
    #     task='classification',
    #     model_name='pepbert'
    # )
    # rf/svm/xgb/lstm/transformer/esm
    val_dataset = PepDataset(
            sequences=X_val,
            labels=y_val,
            task=args.task,
            max_len=args.max_len,
            model_name=args.model
        )

    if args.model in ['transformer', 'lstm', 'esm', 'pepbert']:
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    else:
        features_train = train_dataset.features
        features_val = val_dataset.features

    if args.model in ['rf', 'svn', 'xgb']:
        input_dim = 20 # onehot，其他模型不使用
    else:
        input_dim = None

    # === 2. 模型管理器 ===
    manager = ModelManager()
    # 2.1 加载或创建模型
    if args.resume:
        model = manager.load_model(
            path=args.resume,
            name=args.model,
            task=args.task,
            max_len=args.max_len,
            input_dim=input_dim,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            random_state=args.random_state
        )
        print(f"[INFO] Resumed model from {args.resume}")
    else:
        model = manager.create_model(
            name=args.model,
            task=args.task,
            max_len=args.max_len,
            input_dim=input_dim,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            random_state=args.random_state
        )
    # esm/pepbert
    # predicts_model = PredictModel(hidden_size=480) # esm:480
    # predicts_model = PredictModel(hidden_size=1024) # prot_bert:1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # predicts_model = predicts_model.to(device) # esm/pepbert
    model = model.to(device) # lstm/transformer/esm/pepbert

    # 2.2 冻结预训练参数
    if args.freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
        print("[INFO] Pre-trained backbone frozen, only classifier will be trained")

    # === 3. 训练 ===
    # 3.1 初始化最佳指标
    best_val_accuracy = 0.0 if args.task == 'classification' else float('inf')
    best_val_loss = float('inf')
    best_epoch = 0

    # 3.2 最佳模型保存路径
    if args.model in ['rf', 'svm', 'xgb']:
        best_model_path = os.path.join(
            args.output_dir,
            f"BEST_{args.model}_{args.task}_{args.data_name}_seed{args.random_state}.pkl"
            # f"BEST_desc_{args.model}_{args.task}_{args.data_name}_seed{args.random_state}.pkl"
        )
    else:
        best_model_path = os.path.join(
            args.output_dir,
            f"BEST_{args.model}_{args.task}_{args.data_name}_seed{args.random_state}.pt"
        )
    # 3.3 训练过程
    if args.model in ['rf', 'svm', 'xgb']:
        # 传统机器学习模型
        print("Training traditional machine learning model...")
        model.fit(features_train, y_train)
        if args.task == 'classification':
            # 分类任务：预测概率
            val_predictions = model.predict(features_val)
            train_predictions = model.predict(features_train)

            # 计算准确率
            val_accuracy = accuracy_score(y_val, val_predictions > 0.5)
            train_accuracy = accuracy_score(y_train, train_predictions > 0.5)

            # 计算损失
            train_loss = log_loss(y_train, train_predictions)
            val_loss = log_loss(y_val, val_predictions)

        else:
            # 回归任务：直接预测
            val_predictions = model.predict(features_val)
            train_predictions = model.predict(features_train)

            # 计算MSE损失
            train_loss = mean_squared_error(y_train, train_predictions)
            val_loss = mean_squared_error(y_val, val_predictions)

            # 回归任务没有准确率概念
            val_accuracy = None
            train_accuracy = None

        # 记录到wandb
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

        # 打印结果
        if args.task == 'classification':
            print(f"[Final] Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        else:
            print(f"[Final] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最佳模型（传统模型只有一次训练）
        joblib.dump(model, best_model_path)
        print(f"✅ Best model saved to: {best_model_path}")
    else:
        # 深度学习模型
        patience = 100
        no_improve_count = 0
        criterion = nn.BCELoss() if args.task == 'classification' else nn.MSELoss()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, predicts_model.parameters()), lr=1e-3, weight_decay=1e-3) # esm/pepbert
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,weight_decay=1e-3) # lstm/transformer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)

        for epoch in range(500):
            # ===== 训练阶段 =====
            # esm/pepbert
            # predicts_model.train()
            # model.eval()
            model.train() # lstm/transformer

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in train_loader:
                optimizer.zero_grad()
                seqs, labels = batch
                labels = labels.to(device)
                # esm/pepbert
                # embed = seqs.to(device)# pepbert_embed
                # embed = model(sequences=seqs)
                # outputs = predicts_model(embed)
                outputs = model(seqs) # lstm/transformer
                outputs = outputs.squeeze()


                if args.task == 'classification':
                    labels = labels.float()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 计算训练准确率
                if args.task == 'classification':
                    with torch.no_grad():
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        total_correct += correct
                        total_samples += labels.size(0)

            # 计算平均指标
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = total_correct / total_samples if args.task == 'classification' else None

            # 学习率调度
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # current_lr = 0.001

            # ===== 验证阶段 =====
            # esm/pepbert
            # predicts_model.eval()
            model.eval() # lstm/transformer
            val_loss = 0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    seqs, labels = batch
                    # esm/pepbert
                    # embed = model(sequences=seqs)
                    # embed = seqs.to(device) #pepbert_embed
                    # outputs = predicts_model(embed)
                    outputs = model(seqs)  # lstm/transformer
                    labels = labels.to(device)
                    outputs = outputs.squeeze()

                    if args.task == 'classification':
                        labels = labels.float()

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # 计算验证准确率
                    if args.task == 'classification':
                        predictions = (outputs > 0.5).float()
                        correct = (predictions == labels).sum().item()
                        val_correct += correct
                        val_samples += labels.size(0)

            # 计算平均验证指标
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_samples if args.task == 'classification' else None

            # ===== 记录到wandb =====
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

            # ===== 打印信息 =====
            if args.task == 'classification':
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # ===== 保存最佳模型 =====
            if args.task == 'classification':
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path) # lstm/transformer
                    # torch.save(predicts_model.state_dict(), best_model_path) # esm/pepbert
                    no_improve_count = 0
                    print(f"✅ New best model saved with val accuracy: {val_accuracy:.4f}")
                else:
                    no_improve_count += 1
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path) # lstm/transformer
                    # torch.save(predicts_model.state_dict(), best_model_path) # esm/pepbert
                    no_improve_count = 0
                    print(f"✅ New best model saved with val loss: {avg_val_loss:.4f}")
                else:
                    no_improve_count += 1

            # ===== 早停检查 =====
            if no_improve_count >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

        # ===== 训练结束后 =====
        print(f"\n Training completed! Best model from epoch {best_epoch}")
        if args.task == 'classification':
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"Best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Best validation loss: {best_val_loss:.4f}")

    # === 4. 关闭 wandb ===
    if use_wandb:
        wandb.finish()
if __name__ == '__main__':
    main()
