from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    """
    - essay_texts: ['에세이1', '에세이2', ...]
    - scores: [[s1_1, s1_2, ..., s1_11], [s2_1, ..., s2_11], ...] (각 에세이별 11개 점수)
    - tokenizer: Transformers 토크나이저
    - max_length: 한 번에 입력할 최대 토큰 길이
    """
    def __init__(self, essay_texts, scores, tokenizer, max_length=4096):
        self.essay_texts = essay_texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essay_texts)

    def __getitem__(self, idx):
        essay = self.essay_texts[idx]
        label = torch.tensor(self.scores[idx], dtype=torch.float)

        # 여기에서 Prompt-as-Prefix 구조를 간단히 적용할 수 있음
        # 예: "채점 지시문 + 에세이 텍스트" 형태로 구성
        # 실제로는 더 복잡한 Prompt를 구성해도 됨

        encoding = self.tokenizer(
            essay, 
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Pytorch DataLoader에서 사용될 수 있도록 dict로 반환
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }
        return item