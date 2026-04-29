import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.prepro_tweet_cleaner import clean_tweet

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH       = "data/sentiment_10k.csv"
TFIDF_PATH      = "models/tfidf_lr.joblib"
DISTILBERT_PATH = "models/distilbert_sentiment"
DEVICE          = torch.device("mps")
LABEL_MAP       = {0: "negative", 1: "positive"}
N_SAMPLES       = 500
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("🐦 Twitter Sentiment Analysis — Error Analysis Dashboard")
st.caption("Comparing TF-IDF + Logistic Regression vs DistilBERT")

# ── Load models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf = joblib.load(TFIDF_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
    db_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    db_model.to(DEVICE)
    db_model.eval()
    return tfidf, tokenizer, db_model

tfidf_model, db_tokenizer, db_model = load_models()

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "sentiment"]].dropna()
    df["sentiment"] = df["sentiment"].astype(int)
    df["clean_text"] = df["text"].astype(str).apply(clean_tweet)
    return df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

df = load_data()

# ── Run predictions ───────────────────────────────────────────────────────
@st.cache_data
def run_predictions():
    # TF-IDF
    tfidf_preds = tfidf_model.predict(df["clean_text"])
    tfidf_proba = tfidf_model.predict_proba(df["clean_text"]).max(axis=1)

    # DistilBERT
    db_preds, db_proba = [], []
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch = df["clean_text"][i:i+batch_size].tolist()
        inputs = db_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = db_model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)
        db_preds.extend(proba.argmax(dim=-1).cpu().tolist())
        db_proba.extend(proba.max(dim=-1).values.cpu().tolist())

    return tfidf_preds, tfidf_proba, db_preds, db_proba

with st.spinner("Running predictions..."):
    tfidf_preds, tfidf_proba, db_preds, db_proba = run_predictions()

df["tfidf_pred"]  = tfidf_preds
df["tfidf_conf"]  = tfidf_proba
df["db_pred"]     = db_preds
df["db_conf"]     = db_proba

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
model_choice = st.sidebar.radio("Select Model", ["TF-IDF + LR", "DistilBERT"])
pred_col = "tfidf_pred" if model_choice == "TF-IDF + LR" else "db_pred"
conf_col = "tfidf_conf" if model_choice == "TF-IDF + LR" else "db_conf"

# ── Benchmark Table ───────────────────────────────────────────────────────
st.subheader("📊 Benchmark Summary")
bench = pd.DataFrame({
    "Model":        ["TF-IDF + LR", "DistilBERT"],
    "Accuracy":     ["76.45%", "80.05%"],
    "Macro F1":     ["0.7644", "0.8003"],
    "Latency":      ["0.010 ms/tweet", "0.525 ms/tweet"]
})
st.table(bench)

# ── Confusion Matrix ──────────────────────────────────────────────────────
st.subheader(f"🔢 Confusion Matrix — {model_choice}")
cm = confusion_matrix(df["sentiment"], df[pred_col])
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ── Confidence Distribution ───────────────────────────────────────────────
st.subheader(f"📈 Confidence Distribution — {model_choice}")
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.hist(df[conf_col], bins=20, color="steelblue", edgecolor="white")
ax2.set_xlabel("Confidence")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# ── Misclassified Examples ────────────────────────────────────────────────
st.subheader(f"❌ Top Misclassified Examples — {model_choice}")
wrong = df[df["sentiment"] != df[pred_col]].copy()
wrong = wrong.sort_values(conf_col, ascending=False).head(10)

for _, row in wrong.iterrows():
    st.markdown(f"""
    **Tweet:** {row['text']}
    - **Actual:** `{LABEL_MAP[row['sentiment']]}` | **Predicted:** `{LABEL_MAP[row[pred_col]]}` | **Confidence:** `{row[conf_col]:.2%}`
    ---
    """)

# ── Live Predictor ────────────────────────────────────────────────────────
st.subheader("🧪 Live Tweet Predictor")
user_input = st.text_area("Enter a tweet:")
if st.button("Predict"):
    cleaned = clean_tweet(user_input)
    if model_choice == "TF-IDF + LR":
        proba = tfidf_model.predict_proba([cleaned])[0]
        label = LABEL_MAP[int(proba.argmax())]
        conf  = proba.max()
    else:
        inputs = db_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = db_model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0]
        label = LABEL_MAP[int(proba.argmax())]
        conf  = float(proba.max())

    st.success(f"**Sentiment:** {label} | **Confidence:** {conf:.2%}")