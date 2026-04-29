import joblib
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.prepro_tweet_cleaner import clean_tweet
import time

# ── Config ───────────────────────────────────────────────────────────────
TFIDF_PATH      = "models/tfidf_lr.joblib"
DISTILBERT_PATH = "models/distilbert_sentiment"
DEVICE          = torch.device("mps")
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Tweet Sentiment API", version="1.0")

# ── Load models at startup ────────────────────────────────────────────────
print("Loading TF-IDF model...")
tfidf_model = joblib.load(TFIDF_PATH)

print("Loading DistilBERT model...")
db_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
db_model     = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
db_model.to(DEVICE)
db_model.eval()
print("Models ready.")
# ─────────────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "negative", 1: "positive"}


class PredictRequest(BaseModel):
    text: str
    model: str = "distilbert"  # "tfidf" or "distilbert"


class PredictResponse(BaseModel):
    text: str
    model_used: str
    sentiment: str
    confidence: float
    latency_ms: float


def predict_tfidf(text: str):
    cleaned = clean_tweet(text)
    start = time.perf_counter()
    proba = tfidf_model.predict_proba([cleaned])[0]
    latency = (time.perf_counter() - start) * 1000
    label = int(proba.argmax())
    return LABEL_MAP[label], round(float(proba.max()), 4), round(latency, 3)


def predict_distilbert(text: str):
    cleaned = clean_tweet(text)
    inputs = db_tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding=True, max_length=64
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    start = time.perf_counter()
    with torch.no_grad():
        logits = db_model(**inputs).logits
    latency = (time.perf_counter() - start) * 1000
    proba = torch.softmax(logits, dim=-1)[0]
    label = int(proba.argmax())
    return LABEL_MAP[label], round(float(proba.max()), 4), round(latency, 3)


@app.get("/health")
def health():
    return {"status": "ok", "models_available": ["tfidf", "distilbert"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model == "tfidf":
        sentiment, confidence, latency = predict_tfidf(req.text)
    else:
        sentiment, confidence, latency = predict_distilbert(req.text)

    return PredictResponse(
        text=req.text,
        model_used=req.model,
        sentiment=sentiment,
        confidence=confidence,
        latency_ms=latency
    )


@app.post("/batch_predict")
def batch_predict(texts: list[str], model: str = "distilbert"):
    results = []
    for text in texts:
        if model == "tfidf":
            sentiment, confidence, latency = predict_tfidf(text)
        else:
            sentiment, confidence, latency = predict_distilbert(text)
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "latency_ms": latency
        })
    return results