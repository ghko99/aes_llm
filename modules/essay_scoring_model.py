import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch

class AESModel(nn.Module):
    def __init__(self, model_name_or_path="Bllossom/llama-3.2-Korean-Bllossom-3B", num_scores=11, hidden_size=768):
        """
        model_name_or_path: 백본으로 사용할 LLM의 경로/이름
        num_scores: 예측해야 할 점수(항목) 개수
        hidden_size: LLM의 마지막 hidden state 크기 (모델에 따라 조정 필요)
        """
        super().__init__()
        # 2.1) Pretrained Language Model (Frozen)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.backbone  = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False  # LLM 파라미터 동결

        # 2.2) Projection Layer (Linear)
        self.linear = nn.Linear(hidden_size, num_scores)

    def forward(self, input_ids, attention_mask):
        # 1) Backbone LLM Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # 2) Flatten/Pooling 등으로 하나의 벡터 추출
        # 예시: [CLS] 토큰(또는 첫 번째 토큰)의 hidden state를 대표로 사용 (모델별로 다를 수 있음)
        # GPT 계열은 공식 CLS 토큰이 없으니, 그냥 첫 번째 토큰 혹은 평균 풀링 등 원하는 방식 사용
        # 여기서는 간단히 '평균 풀링(mean pooling)' 예시
        pooled = torch.mean(last_hidden_state, dim=1)  # (batch_size, hidden_size)

        # 3) Projection Layer 통과 (11차원 예측)
        scores = self.linear(pooled)  # (batch_size, num_scores)

        return scores
