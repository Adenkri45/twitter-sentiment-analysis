# lstm_model.py - Updated for Sampled Data

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def run_lstm_model(sample_csv_path, max_words=10000, max_len=100, embedding_dim=100):
    # Load sampled dataset
    df = pd.read_csv(sample_csv_path, encoding='latin-1')

    # Clean column names if needed
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

    # Preprocessing: Normalize Sentiment
    df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})  # binary labels: 0 = negative, 1 = positive

    X = df['text'].values
    y = df['sentiment'].values

    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary output

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train model
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Save model and tokenizer
    model.save('models/lstm_sentiment_model.h5')
    with open('models/lstm_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("\n✅ Model and tokenizer saved successfully!")

    # ✅ NEW: Save results + return acc
    from src.utils import save_and_return_results
    acc = save_and_return_results(y_test, y_pred, model_name="lstm")

    return acc, model, tokenizer
