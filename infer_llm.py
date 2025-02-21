import torch
import torch.nn as nn
from modules.essay_scoring_model import AESModel
from transformers import AutoTokenizer
from torch.amp import autocast

def load_trained_aes_model(
    model_name="Bllossom/llama-3.2-Korean-Bllossom-3B",
    checkpoint_path="./checkpoints/linear_best.pth",
    hidden_size=3072,  # 모델 설정에 맞춰 조정
    num_scores=11
):
    """
    저장된 Linear 레이어 가중치를 로드해 AESModel을 복원하는 함수
    """
    # 1) 모델 초기화
    model = AESModel(
        model_name_or_path=model_name, 
        hidden_size=hidden_size, 
        num_scores=num_scores
    ).cuda()
    # 2) 저장된 linear 가중치 불러오기
    state_dict = torch.load(checkpoint_path)
    model.linear.load_state_dict(state_dict)
    print(f"Loaded linear layer weights from {checkpoint_path}")

    # 평가 모드
    model.eval()
    return model

def inference_essay(model, tokenizer, text, max_length=512):
    """
    단일 에세이 텍스트(문자열)에 대해 추론하여 점수를 반환
    """
    # 1) 토크나이징
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # 2) 모델 추론
    
    with torch.no_grad():
        with autocast(device_type='cuda'):
            preds = model(input_ids.cuda(), attention_mask.cuda())  # (1, num_scores)
    
    # 텐서를 파이썬 리스트나 numpy 등으로 변환
    scores = preds.squeeze(0).tolist()
    return scores

if __name__ == "__main__":
    # 1) 모델 & 토크나이저 로드
    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 필요 시 설정
    model = load_trained_aes_model(
        model_name=model_name,
        checkpoint_path="./checkpoints/linear_best.pth",
        hidden_size=3072,  # 학습할 때 사용한 값
        num_scores=11
    )

    # 2) 테스트용 에세이(문자열)
    test_essay = """\
스마트폰 사용을 통해 시력이 나빠진 경험을 서술한 에세이입니다.
어렸을 때부터 스마트폰에 과도하게 의존하여 글을 읽을 기회가 줄어들었고, 
결과적으로 시력이 급격히 떨어졌습니다. 
이를 계기로 시력 관리의 중요성을 깨닫게 되었다는 내용입니다.
"""

    # 3) 단일 에세이 Inference
    predicted_scores = inference_essay(model, tokenizer, test_essay)
    print("[Inference] Predicted scores:", predicted_scores)

