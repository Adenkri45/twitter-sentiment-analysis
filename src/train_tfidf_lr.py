import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.prepro_tweet_cleaner import clean_tweet


DATA_PATH = "data/sentiment_10k.csv"   # change if needed
TEXT_COL = "text"                               # change if needed
LABEL_COL = "sentiment"                             # change if needed
OUT_DIR = "models"
MODEL_PATH = os.path.join(OUT_DIR, "tfidf_lr.joblib")


def main():
    print(">>> Training script started")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].replace({4: 1})

    df["clean_text"] = df[TEXT_COL].astype(str).apply(clean_tweet)

    X = df["clean_text"]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200000,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
