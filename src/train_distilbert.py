import os
import time
import joblib
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH  = "data/sentiment_10k.csv"
TEXT_COL   = "text"
LABEL_COL  = "sentiment"
OUT_DIR    = "models"
MODEL_PATH = os.path.join(OUT_DIR, "distilbert_sentiment")

BATCH_SIZE = 32
EPOCHS     = 3
MAX_LEN    = 64
LR         = 2e-5
DEVICE     = torch.device("mps")  # M4 Metal
# ────────────────────────────────────────────────────────────────────────


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    return all_preds, all_labels


def main():
    print(">>> DistilBERT training started")
    print(f">>> Device: {DEVICE}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL_COL],
        test_size=0.2, random_state=42, stratify=df[LABEL_COL]
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = TweetDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset  = TweetDataset(X_test.tolist(),  y_test.tolist(),  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start = time.time()

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | Loss: {total_loss/(i+1):.4f}")

        elapsed = time.time() - start
        print(f"Epoch {epoch+1} done in {elapsed:.1f}s | Avg Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    preds, labels = evaluate(model, test_loader)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, digits=4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(labels, preds))

    # Save
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()