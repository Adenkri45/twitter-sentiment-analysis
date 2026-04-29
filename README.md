# 🐦 Twitter Sentiment Analysis — NLP Benchmarking System

A production-style NLP system that classifies tweet sentiment (positive/negative) by benchmarking classical ML against transformer models under constrained compute.

## 🎯 Project Narrative

> "Accuracy vs latency vs cost under constrained compute"

Instead of just training one model, this project compares a classical ML pipeline against a transformer, measures both on accuracy, F1, and inference latency, and deploys the best model behind a REST API with an error analysis dashboard.

---

## 📊 Benchmark Results

| Model         | Accuracy | Macro F1 | Latency (ms/tweet) |
|---------------|----------|----------|--------------------|
| TF-IDF + LR   | 76.45%   | 0.7644   | 0.010              |
| DistilBERT    | 80.05%   | 0.8003   | 0.525              |

- DistilBERT achieves **+3.6% accuracy** over the classical baseline
- TF-IDF + LR is **53x faster** — better for latency-sensitive applications
- Models trained on Apple M4 MPS (Metal GPU) — no cloud GPU required

---

## 🏗️ Project Structure

├── src/
│   ├── prepro_tweet_cleaner.py   # Tweet normalizer (URLs, emojis, hashtags, negation)
│   ├── train_tfidf_lr.py         # TF-IDF + Logistic Regression training
│   ├── train_distilbert.py       # DistilBERT fine-tuning (MPS optimized)
│   └── benchmark_latency.py      # Accuracy vs latency benchmarking
├── api/
│   └── main.py                   # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py          # Error analysis dashboard
└── main.py                       # Original v1 (Naive Bayes + LSTM + BERT)

---

## 🚀 Setup

```bash
git clone https://github.com/Adenkri45/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Running the Project

### 1. Train Baseline (TF-IDF + LR)
```bash
python -m src.train_tfidf_lr
```

### 2. Train DistilBERT
```bash
python -m src.train_distilbert
```

### 3. Run Latency Benchmark
```bash
python -m src.benchmark_latency
```

### 4. Start FastAPI
```bash
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🔍 Key Features

- **Tweet Preprocessing** — handles URLs, mentions, hashtags, emojis, elongated words, negation
- **Model Benchmarking** — accuracy, macro F1, per-class metrics, inference latency
- **REST API** — `/predict`, `/batch_predict`, `/health` with model switching
- **Error Analysis Dashboard** — confusion matrix, confidence distribution, top misclassified examples, live predictor
- **Constrained Compute** — trained on 10k sample, runs fully local on Apple Silicon

---

## 📦 Dataset

[Sentiment140](http://help.sentiment140.com/) — 1.6M tweets labelled positive/negative. Stratified 10k sample used for training.

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `HuggingFace Transformers` `PyTorch` `FastAPI` `Streamlit` `pandas` `Apple MPS`
