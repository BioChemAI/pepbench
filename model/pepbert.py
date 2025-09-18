import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List, Union


class PepBERTModel(nn.Module):
    def __init__(
            self,
            model_path: str = "MODEL/prot_bert",
            max_len: int = None,
            task: str = "classification",
            device: torch.device = None,
    ):
        """
        PepBERT 模型封装类，用于多肽序列的特征提取。
        :param model_path: 本地路径
        :param max_len: 最大序列长度
        :param device: 指定运行设备
        """
        super().__init__()

        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        # 加载 tokenizer 和预训练模型
        self.pepbert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.pepbert.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

        self.to(self.device)

    def forward(
            self,
            sequences: Union[List[str], torch.Tensor] = None,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            last: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """
        :param sequences: 多肽序列列表（每个元素是一个氨基酸序列字符串）
        :param input_ids: token ids
        :param attention_mask: attention mask
        :param last: 是否使用[CLS] token作为输出
        """
        if sequences is not None:
            spaced_sequences = [" ".join(list(seq)) for seq in sequences]
            inputs = self.tokenizer(
                spaced_sequences,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_len
            ).to(self.device)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 如果传入的是 HuggingFace 风格的张量
        elif input_ids is not None:
            inputs = {"input_ids": input_ids.to(self.device)}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask.to(self.device)
        else:
            raise ValueError("forward() 需要 sequences (List[str]) 或 input_ids 张量")

        # 模型推理
        with torch.no_grad():
            outputs = self.pepbert(**inputs)

        # 特征提取
        if last:
            # 使用 [CLS] token 作为整个序列的嵌入
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # 使用平均池化（排除特殊token）
            last_hid = outputs.last_hidden_state
            mask = inputs['attention_mask'].clone()

            # 对于BERT，需要排除[CLS]、[SEP]、[PAD]等特殊token
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # 将特殊token的mask设为0
            special_tokens = [cls_token_id, sep_token_id, pad_token_id]
            for i, seq in enumerate(inputs['input_ids']):
                for j, token_id in enumerate(seq):
                    if token_id in special_tokens:
                        mask[i, j] = 0

                mask_expand = mask.unsqueeze(-1).expand(last_hid.size())
                sum_hid = (last_hid * mask_expand).sum(1)
                lengths = mask.sum(1).unsqueeze(-1).clamp(min=1)
                embedding = sum_hid / lengths

        return embedding
