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
