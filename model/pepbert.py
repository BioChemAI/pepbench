# model/pepbert_model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List


class PepBERTModel(nn.Module):
    def __init__(
        self,
        model_path: str = "MODEL/prot_bert",  # 本地模型路径
        task: str = "classification",
        device: torch.device = None,
    ):
        """
        PepBERT 模型封装类，用于多肽序列的分类或回归任务。

        :param model_path: 本地路径或 Huggingface Hub ID
        :param task: "classification" or "regression"
        :param device: 指定运行设备
        """
        super().__init__()
        self.task = task
        self.hidden_size = 1024  
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 tokenizer 和预训练模型（可从本地路径加载）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
        self.pepbert = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)

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
        # PepBERT 通常需要将序列转成用空格分开的 amino acids
        spaced_seqs = [" ".join(list(seq)) for seq in sequences]

        inputs = self.tokenizer(
            spaced_seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # 通常小于 BERT 限制
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.pepbert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token 的表示
        logits = self.output_layer(cls_embedding)

        if self.task == 'classification':
            return logits.squeeze(1)  # [batch_size]
        else:
            return logits             # [batch_size, 1]
