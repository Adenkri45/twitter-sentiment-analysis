import os
import pandas as pd
import time
import nltk
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from src.preprocessing import clean_tweet
from src.visualize import plot_accuracy_f1, plot_confusion_matrix
from src.utils import save_and_return_results


# ------------------- Load and preprocess full dataset ------------------- #

# Load full dataset (no header row)
df = pd.read_csv('data/training_1600000_processed_noemoticon.csv', encoding='latin-1', header=None)
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Filter only 0 (negative) and 4 (positive)
df = df[df['sentiment'].isin([0, 4])]

# Remap sentiment to binary: 0 = negative, 1 = positive
df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})

# Clean text
df['clean_text'] = df['text'].apply(clean_tweet)

# Sanity check
if df.empty:
    raise ValueError("❌ ERROR: Preprocessed DataFrame is empty. Check filtering.")
else:
    print(f"✅ Preprocessed DataFrame ready. Shape: {df.shape}")
    print("Class distribution:", df['sentiment'].value_counts())


# ------------------- Load and preprocess dataset ------------------- #

# Load small sample safely (file already has headers)
#df = pd.read_csv('data/sentiment_sample_small.csv', encoding='latin-1')

# Filter only 0, 2, 4 labels
#df = df[df['sentiment'].isin([0, 2, 4])]

# Replace label values (optional: remap neutral to 1)
#df['sentiment'] = df['sentiment'].replace({0: 0, 2: 1, 4: 2})

# Clean text
#df['clean_text'] = df['text'].apply(clean_tweet)

# Validate DataFrame
#if df.empty:
    #raise ValueError("❌ ERROR: Preprocessed DataFrame is empty. Check class filtering or sampling.")
#else:
    #print("✅ Preprocessed DataFrame ready. Shape:", df.shape)
    #print("Label Distribution After Remapping:")
    #print(df['sentiment'].value_counts())
    #print("Unique labels:", df['sentiment'].unique())


#if df.empty:
    #raise ValueError("❌ ERROR: Preprocessed DataFrame is empty. Check class filtering or sampling.")
#else:
    #print(f"✅ Preprocessed DataFrame ready. Shape: {df.shape}")

#print("Class distribution:", df['sentiment'].value_counts())

# ----------------- MODEL TRAINING FUNCTIONS ----------------- #

def train_naive_bayes(df):
    from src.naive_bayes import run_naive_bayes
    print("\n Training Naive Bayes Model...\n")
    start = time.time()
    acc = run_naive_bayes(df)
    end = time.time()
    print(f"⏱ Naive Bayes Training Time: {end - start:.2f} seconds\n")

def train_lstm():
    from src.lstm_model import run_lstm_model
    print("\n Training LSTM Model...\n")
    start = time.time()
    acc, model, tokenizer = run_lstm_model('data/training_1600000_processed_noemoticon.csv')
    end = time.time()
    print(f"⏱ LSTM Training Time: {end - start:.2f} seconds\n")

def train_bert():
    from src.bert_model import run_bert_model
    print("\n Training BERT Model...\n")
    start = time.time()
    acc, model, tokenizer = run_bert_model('data/training_1600000_processed_noemoticon.csv', epochs=1)
    end = time.time()
    print(f"⏱️ BERT Training Time: {end - start:.2f} seconds\n")

# ----------------- TRAIN ALL MODELS ----------------- #
#  Automatically train all models
nb_acc = train_naive_bayes(df)
lstm_acc = train_lstm()
bert_acc = train_bert()

# --------models = ['Naive Bayes', 'LSTM', 'BERT']
os.makedirs("results", exist_ok=True)
accuracies = [nb_acc, lstm_acc, bert_acc]  # collected from each model

plt.bar(models, accuracies, color='skyblue')
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.savefig("results/model_accuracy_comparison.png")
plt.show()

print("✅ Model accuracy comparison saved to results/model_accuracy_comparison.png")