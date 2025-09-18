import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Union


class ESMModel(nn.Module):
    def __init__(
            self,
            model_path: str = "MODEL/esm2_t12_35M_UR50D",
            max_len: int = None,
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

        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        # 加载 tokenizer 和预训练模型
        self.esm = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.hidden_size = self.esm.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)


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
        :param input_ids: token ids (HuggingFace 风格调用)
        :param attention_mask: attention mask (HuggingFace 风格调用)
        """
        # 如果传入的是 List[str]，先做 tokenizer 编码
        with torch.no_grad():
            if sequences is not None:
                inputs = self.tokenizer(
                    sequences,
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

            # ESM backbone
            outputs = self.esm(**inputs)
            if last:
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                last_hid = outputs.last_hidden_state
                mask = inputs['attention_mask'].clone()
                cls,eos,pad = self.tokenizer.cls_token_id,self.tokenizer.eos_token_id,self.tokenizer.pad_token_id
                for i,seq in enumerate(inputs['input_ids']):
                    for j,token_id in enumerate(sequences):
                        if token_id == [cls,eos,pad]:
                            mask[i,j]=0

                mask_expand = mask.unsqueeze(-1).expand(last_hid.size())
                sum_hid = (last_hid*mask_expand).sum(1)
                lengths = mask.sum(1).unsqueeze(-1).clamp(min=1)
                embedding = sum_hid/lengths
            return embedding
