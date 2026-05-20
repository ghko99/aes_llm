# Run Notes

This repository uses separate scripts for training, inference, and evaluation. Keep the handoff between those steps explicit.

## Suggested Flow

1. Install dependencies from `requirements.txt`.
2. Confirm the dataset path expected by `train_llm.py`.
3. Run training and record the checkpoint path.
4. Run `infer_llm.py` with that checkpoint.
5. Run `performance.py` on the generated predictions.

## Metadata To Save

- Git commit and command lines.
- Base model or checkpoint path.
- Dataset split and preprocessing version.
- Training hyperparameters.
- Prediction output path and metric output path.

This makes it easier to distinguish model changes from inference or metric-script changes.
