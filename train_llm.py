import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from modules.essay_dataset import EssayDataset
from modules.essay_scoring_model import AESModel
from torch.amp import autocast, GradScaler
import os
import logging
import sys

def train_essay_llm():
    dataset_from_disk = load_from_disk("aes_dataset")

    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = EssayDataset(dataset_from_disk['train']['text'][:10], 
                                 dataset_from_disk['train']['label'][:10], 
                                 tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    valid_dataset = EssayDataset(
        dataset_from_disk['test']['text'][:5],
        dataset_from_disk['test']['label'][:5],
        tokenizer=tokenizer,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    model = AESModel(model_name_or_path=model_name).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.linear.parameters(), lr=1e-5, weight_decay=0.01)

    # AMP for mixed precision training
    scaler = GradScaler(device='cuda')

    num_epochs = 10


    os.makedirs("./checkpoints", exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_steps = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].half().cuda()

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            train_steps += 1
            logger.info(f"[Step {train_steps}/{len(train_dataloader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / train_steps if train_steps > 0 else 0.0
        model.eval()

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].half().cuda()
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    preds = model(input_ids, attention_mask)
                    loss = criterion(preds, labels)

                total_loss += loss.item()
                count += 1

        avg_valid_loss = total_loss / count if count > 0 else 0.0

        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            checkpoint_path = f"./checkpoints/linear_best.pth"
            torch.save(model.linear.state_dict(), checkpoint_path)
            logger.info(f"=> Best Valid Loss: {best_val_loss:.4f} at epoch {best_epoch}, saved linear weights to {checkpoint_path}")

    return model, tokenizer

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
    )
    logger = logging.getLogger(__name__)

    train_essay_llm()


