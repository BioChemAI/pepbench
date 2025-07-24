# model/esm_model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List


class ESMModel(nn.Module):
    def __init__(
        self,
        model_path: str = "MODEL/esm2_t12_35M_UR50D",
        task: str = "classification",
        device: torch.device = None,
    ):
        """
        ESM 模型封装类，用于多肽序列的分类或回归任务。

        :param model_path: 本地路径或 Huggingface Hub ID
        :param task: "classification" or "regression"
        :param device: 指定运行设备
        """
        super().__init__()
        self.task = task
        self.hidden_size = 480  # 对应 esm2_t12_35M_UR50D
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 tokenizer 和预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.esm = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)

        # 定义输出层
        if task == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            )
        elif task == 'regression':
            self.output_layer = nn.Linear(self.hidden_size, 1)
        else:
            raise ValueError("task must be either 'classification' or 'regression'")

        self.to(self.device)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        :param sequences: 多肽序列列表（每个元素是一个氨基酸序列字符串）
        :return: 模型输出，shape = [batch_size] for classification, [batch_size, 1] for regression
        """
        inputs = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.esm(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        logits = self.output_layer(cls_embedding)            # [batch_size, 1]

        if self.task == 'classification':
            return logits.squeeze(1)  # [batch_size]，兼容 BCEWithLogitsLoss
        else:
            return logits             # [batch_size, 1]，适用于回归任务

