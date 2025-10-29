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
        PepBERT A model encapsulation class used for feature extraction of peptide sequences.
        :param model_path: Pretrained model local path
        :param max_len: Maximum sequence length
        :param device: Running device
        """
        super().__init__()

        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        # Load the tokenizer and the pre-trained model
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
        :param sequences: List of peptide sequences
        :param input_ids: token ids
        :param attention_mask: attention mask
        :param last: List of polypeptide sequences
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

        # If the input is a HuggingFace-style tensor
        elif input_ids is not None:
            inputs = {"input_ids": input_ids.to(self.device)}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask.to(self.device)
        else:
            raise ValueError("forward() 需要 sequences (List[str]) 或 input_ids 张量")

        # Model inference
        with torch.no_grad():
            outputs = self.pepbert(**inputs)

        # Feature extraction
        if last:
            # Use the [CLS] token as the embedding of the entire sequence
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # Using average pooling (excluding special tokens)
            last_hid = outputs.last_hidden_state
            mask = inputs['attention_mask'].clone()

            # For BERT, it is necessary to exclude special tokens such as [CLS], [SEP], [PAD], etc.
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # Set the mask of the special token to 0
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
