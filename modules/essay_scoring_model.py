import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

class AESModel(nn.Module):
    def __init__(self, 
                 model_name_or_path="Bllossom/llama-3.2-Korean-Bllossom-3B",
                 num_scores=11,
                 hidden_size=3072):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # LLM 실치 hidden_size=3200과 동일하게 설정
        self.linear = nn.Linear(hidden_size, num_scores)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]        # (batch, seq_len, 3200)
        last_token_emb = last_hidden_state[:, -1, :]         # (batch, 3200)

        # 여기서 mean(dim=1)을 사용하면 안 됩니다.
        # last_token_emb는 이미 (batch, hidden_size)
        scores = self.linear(last_token_emb)  # (batch, num_scores)

        return scores
