# aes_llm

LLM-based Korean automated essay scoring experiment code.

## Repository layout

- `train_llm.py`: trains the scoring model and saves checkpoints/logs.
- `infer_llm.py`: runs inference with a trained checkpoint.
- `performance.py`: computes evaluation metrics from predictions.
- `modules/`: dataset, model, loss, and utility modules used by the training scripts.

## Setup

```bash
pip install -r requirements.txt
```

Prepare the dataset path expected by the training script, then run:

```bash
python train_llm.py
```

Use `infer_llm.py` after training to generate predictions for evaluation.
