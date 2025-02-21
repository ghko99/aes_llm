import torch
import torch.nn as nn
from modules.essay_scoring_model import AESModel
from transformers import AutoTokenizer
from modules.essay_dataset import EssayDataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_from_disk
from tqdm import tqdm
import json

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

def load_test_dataset():
    dataset_from_disk = load_from_disk("aes_dataset")
    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    test_dataset = EssayDataset(
        dataset_from_disk['test']['text'],
        dataset_from_disk['test']['label'],
        tokenizer=tokenizer,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return test_dataloader

def get_predicted_score():

    model = load_trained_aes_model()
    test_dataloader = load_test_dataset()
    predicted = []
    real_score = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"]
            with autocast(device_type='cuda'):
                preds = model(input_ids, attention_mask)
            scores = preds.squeeze(0).tolist()
            real = labels.squeeze(0).tolist()
            predicted.extend(scores)
            real_score.extend(real)
    return predicted, real_score

if __name__ == "__main__":
    predicted, real_score = get_predicted_score()
    result = {"pred" : predicted, "real":real_score}
    with open('./res.json','w',encoding='cp949') as f:
        json.dump(result,f,indent='\t')