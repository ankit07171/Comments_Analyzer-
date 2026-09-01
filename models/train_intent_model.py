"""
Train Combined Sentiment + Intent Classification Model from CSV Data

Trains ONE shared BERT encoder with TWO classification heads:
  - Sentiment head  -> positive / neutral / negative
  - Intent head     -> complaint, refund_risk, support_request, feature_request,
                        purchase_intent, praise, misinformation_risk, spam,
                        general_discussion (or whatever intents are in your CSV)

This replaces running train_sentiment_from_csv.py and train_intent_from_csv.py
separately. Both heads share the same BERT backbone, so you train once and
get both predictions from a single forward pass.

Usage:
    python train_multitask_from_csv.py --csv_path models/comments_sentiment.csv

Expected CSV columns (defaults, override with flags):
    comment_text : the comment text
    sentiment    : "positive" | "neutral" | "negative"
    intent       : e.g. "complaint", "praise", "spam", ...

Output (in --output_dir, default models/multitask_bert/):
    multitask_model.pt       - trained model weights (backbone + both heads)
    tokenizer/                - BERT tokenizer
    sentiment_labels.json    - sentiment label mapping
    intent_labels.json       - intent label mapping
    training_history.csv     - per-epoch loss/accuracy/F1 for both tasks
    validation_metrics.json  - final classification reports + confusion matrices
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(csv_path: str, text_col: str, sentiment_col: str, intent_col: str):
    df = pd.read_csv(csv_path)

    required = {text_col, sentiment_col, intent_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. Available columns: {list(df.columns)}"
        )

    df = df[[text_col, sentiment_col, intent_col]].copy()
    df[text_col] = df[text_col].map(clean_text)
    df[sentiment_col] = df[sentiment_col].astype(str).str.strip().str.lower()
    df[intent_col] = df[intent_col].astype(str).str.strip().str.lower()

    df = df.dropna().drop_duplicates(subset=[text_col])
    df = df[df[text_col].str.len() > 0]
    df = df[df[sentiment_col].str.len() > 0]
    df = df[df[intent_col].str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remained after cleaning. Check your CSV contents.")

    # Sentiment labels: fixed 3-class scheme (fall back to whatever is present
    # if the CSV uses different sentiment strings)
    default_sentiment_order = ["negative", "neutral", "positive"]
    present_sentiments = sorted(df[sentiment_col].unique())
    sentiment_order = [s for s in default_sentiment_order if s in present_sentiments] or present_sentiments
    sentiment_to_id = {label: i for i, label in enumerate(sentiment_order)}

    # Intent labels: derived from whatever is in the CSV
    intent_order = sorted(df[intent_col].unique())
    intent_to_id = {label: i for i, label in enumerate(intent_order)}

    sent_counts = df[sentiment_col].value_counts()
    intent_counts = df[intent_col].value_counts()

    if sent_counts.min() < 2:
        raise ValueError(
            "Every sentiment class needs at least 2 examples for a stratified split. "
            f"Smallest counts: {sent_counts.to_dict()}"
        )
    if intent_counts.min() < 2:
        raise ValueError(
            "Every intent class needs at least 2 examples for a stratified split. "
            f"Smallest counts: {intent_counts.to_dict()}"
        )

    df["sentiment_label"] = df[sentiment_col].map(sentiment_to_id).astype(int)
    df["intent_label"] = df[intent_col].map(intent_to_id).astype(int)

    print(f"Samples: {len(df)}")
    print(f"Sentiment classes ({len(sentiment_to_id)}):\n{sent_counts.to_string()}")
    print(f"\nIntent classes ({len(intent_to_id)}):\n{intent_counts.to_string()}")

    return df, sentiment_to_id, intent_to_id


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, intent_labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.sentiment_labels = list(sentiment_labels)
        self.intent_labels = list(intent_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
            "intent_label": torch.tensor(self.intent_labels[index], dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Model: shared backbone, two heads
# --------------------------------------------------------------------------- #

class MultiTaskClassifier(nn.Module):
    def __init__(self, num_sentiment_labels: int, num_intent_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)
        self.intent_head = nn.Linear(hidden_size, num_intent_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.sentiment_head(pooled), self.intent_head(pooled)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, loader, device, id_to_sentiment, id_to_intent, output_dir, name):
    model.eval()
    sent_true, sent_pred, intent_true, intent_pred = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_logits, intent_logits = model(input_ids, attention_mask)

            sent_true.extend(batch["sentiment_label"].numpy())
            sent_pred.extend(sentiment_logits.argmax(dim=1).cpu().numpy())
            intent_true.extend(batch["intent_label"].numpy())
            intent_pred.extend(intent_logits.argmax(dim=1).cpu().numpy())

    def summarize(true, pred, id_to_label, task_name):
        label_ids = list(range(len(id_to_label)))
        target_names = [id_to_label[i] for i in label_ids]
        report = classification_report(
            true, pred, labels=label_ids, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        print(f"\n{name.upper()} - {task_name.upper()} RESULTS")
        print(classification_report(
            true, pred, labels=label_ids, target_names=target_names,
            digits=4, zero_division=0
        ))
        cm = confusion_matrix(true, pred, labels=label_ids)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
            output_dir / f"{name}_{task_name}_confusion_matrix.csv"
        )
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    result = {
        "sentiment": summarize(sent_true, sent_pred, id_to_sentiment, "sentiment"),
        "intent": summarize(intent_true, intent_pred, id_to_intent, "intent"),
    }
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, sentiment_to_id, intent_to_id = load_data(
        args.csv_path, args.text_col, args.sentiment_col, args.intent_col
    )
    id_to_sentiment = {v: k for k, v in sentiment_to_id.items()}
    id_to_intent = {v: k for k, v in intent_to_id.items()}

    # Stratify on intent (usually the harder, more imbalanced task)
    train_df, val_df = train_test_split(
        df, test_size=args.validation_size, random_state=args.seed, stratify=df["intent_label"]
    )
    print(f"\nTraining: {len(train_df)}, Validation: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_set = MultiTaskDataset(
        train_df[args.text_col], train_df["sentiment_label"], train_df["intent_label"],
        tokenizer, args.max_length,
    )
    val_set = MultiTaskDataset(
        val_df[args.text_col], val_df["sentiment_label"], val_df["intent_label"],
        tokenizer, args.max_length,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = MultiTaskClassifier(
        num_sentiment_labels=len(sentiment_to_id),
        num_intent_labels=len(intent_to_id),
    ).to(device)

    # Class-balanced loss weights per task
    sentiment_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(sentiment_to_id)),
        y=train_df["sentiment_label"],
    )
    intent_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(intent_to_id)),
        y=train_df["intent_label"],
    )
    sentiment_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(sentiment_weights, dtype=torch.float32, device=device)
    )
    intent_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(intent_weights, dtype=torch.float32, device=device)
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_combined_f1 = -1.0
    history = []

    print("\n" + "=" * 60)
    print("TRAINING (multi-task: sentiment + intent)")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            intent_labels = batch["intent_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            sentiment_logits, intent_logits = model(input_ids, attention_mask)

            # Combined loss: sum of both task losses (weight intent higher
            # since it's usually the harder task with more classes)
            loss = (
                args.sentiment_loss_weight * sentiment_criterion(sentiment_logits, sentiment_labels)
                + args.intent_loss_weight * intent_criterion(intent_logits, intent_labels)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device, id_to_sentiment, id_to_intent, output_dir, "validation")
        combined_f1 = (metrics["sentiment"]["macro_f1"] + metrics["intent"]["macro_f1"]) / 2

        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "sentiment_accuracy": metrics["sentiment"]["accuracy"],
            "sentiment_macro_f1": metrics["sentiment"]["macro_f1"],
            "intent_accuracy": metrics["intent"]["accuracy"],
            "intent_macro_f1": metrics["intent"]["macro_f1"],
            "combined_macro_f1": combined_f1,
        }
        history.append(row)
        print(row)

        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            torch.save(model.state_dict(), output_dir / "multitask_model.pt")
            print("  New best model saved!")

    # Reload best checkpoint and run final evaluation
    model.load_state_dict(torch.load(output_dir / "multitask_model.pt", map_location=device))
    evaluate(model, val_loader, device, id_to_sentiment, id_to_intent, output_dir, "final_validation")

    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "sentiment_labels.json").write_text(
        json.dumps({str(k): v for k, v in id_to_sentiment.items()}, indent=2), encoding="utf-8"
    )
    (output_dir / "intent_labels.json").write_text(
        json.dumps({str(k): v for k, v in id_to_intent.items()}, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    print(f"\n✅ Training complete. Model and artifacts saved to: {output_dir.resolve()}")
    print(f"Best combined macro F1: {best_combined_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="models/comments_sentiment.csv")
    parser.add_argument("--text_col", default="comment_text")
    parser.add_argument("--sentiment_col", default="sentiment")
    parser.add_argument("--intent_col", default="intent")
    parser.add_argument("--output_dir", default="models/multitask_bert")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--sentiment_loss_weight", type=float, default=1.0)
    parser.add_argument("--intent_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())