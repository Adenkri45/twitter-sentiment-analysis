import time
import joblib
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH       = "data/sentiment_10k.csv"
TFIDF_PATH      = "models/tfidf_lr.joblib"
DISTILBERT_PATH = "models/distilbert_sentiment"
DEVICE          = torch.device("mps")
N_SAMPLES       = 200  # tweets to benchmark on
# ─────────────────────────────────────────────────────────────────────────

def benchmark_tfidf(texts):
    model = joblib.load(TFIDF_PATH)
    # warmup
    model.predict(texts[:5])

    start = time.perf_counter()
    model.predict(texts)
    elapsed = time.perf_counter() - start

    ms_per_tweet = (elapsed / len(texts)) * 1000
    return ms_per_tweet


def benchmark_distilbert(texts):
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    model.to(DEVICE)
    model.eval()

    # warmup
    inputs = tokenizer(texts[:5], return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)

    # actual benchmark
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    elapsed = time.perf_counter() - start

    ms_per_tweet = (elapsed / len(texts)) * 1000
    return ms_per_tweet


def main():
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].dropna().astype(str).sample(N_SAMPLES, random_state=42).tolist()

    print(f"Benchmarking on {N_SAMPLES} tweets...\n")

    tfidf_ms = benchmark_tfidf(texts)
    print(f"TF-IDF + LR     : {tfidf_ms:.3f} ms/tweet")

    distilbert_ms = benchmark_distilbert(texts)
    print(f"DistilBERT      : {distilbert_ms:.3f} ms/tweet")

    speedup = distilbert_ms / tfidf_ms
    print(f"\nDistilBERT is {speedup:.1f}x slower than TF-IDF+LR")

    print("\n=== Final Benchmark Table ===")
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Latency (ms/tweet)'}")
    print(f"{'TF-IDF + LR':<20} {'76.45%':<12} {'0.7644':<12} {tfidf_ms:.3f}")
    print(f"{'DistilBERT':<20} {'80.05%':<12} {'0.8003':<12} {distilbert_ms:.3f}")


if __name__ == "__main__":
    main()