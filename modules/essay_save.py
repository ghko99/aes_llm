import pandas as pd
from datasets import Dataset, DatasetDict

def save_prompts(filepath):
    data = {"id":[],"text":[], "label":[]}
    dataset = pd.read_csv(filepath,encoding='utf-8-sig')
    prompt = """
### Instruction:
다음 글(패치들)을 읽고, 아래 11개 평가 항목에 대해 각 항목별 점수를 0~1 사이로 예측하세요.

### Scoring Rubric:
1. 문법의 정확성
2. 단어 사용의 적절성
3. 문장 표현의 적절성
4. 문단 내 구조의 적절성
5. 문단 간 구조의 적절성
6. 구조의 일관성
7. 분량의 적절성
8. 주제의 명료성
9. 사고의 창의성
10. 프롬프트 독해력
11. 설명의 구체성
(총 11개 항목)

### Essay Prompt:
{prompt}

{essay_sentences}
"""

    for d in range(len(dataset)):
        essay_patches = ""
        essay_sentences = dataset.iloc[d]['essay'].split('#@문장구분#')

        for i in range(len(essay_sentences)):
            if essay_sentences[i].strip() != "":
                essay_patches = essay_patches + '### Essay Patch {i}\n{essay_sentence}'.format(i=i+1, essay_sentence=essay_sentences[i])+'\n\n'


        text = prompt.format(prompt = dataset.iloc[d]['essay_prompt'], essay_sentences=essay_patches).strip()

        labels = dataset.iloc[d]['essay_score_avg'].split('#')
        labels = [float(float(l)/3) for l in labels]
        
        data['text'].append(text)
        data['label'].append(labels)
        data['id'].append(dataset.iloc[d]['essay_id'])
    return data

def save_dataset():
    train_data = save_prompts("./dataset/train.csv")
    valid_data = save_prompts('./dataset/valid.csv')
    
    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    total_dataset = DatasetDict({
        "train":train_dataset,
        "test":valid_dataset
    })
    total_dataset.save_to_disk('./aes_dataset')


save_dataset()