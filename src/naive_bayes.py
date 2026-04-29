# naive_bayes.py - Placeholder for Naive Bayes logic

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def run_naive_bayes(df):
    """
    Trains and evaluates a Naive Bayes classifier on preprocessed tweet data.
    df: DataFrame with 'clean_text' and 'sentiment' columns
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_tfidf)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf, vectorizer  # in case you want to save/export later
