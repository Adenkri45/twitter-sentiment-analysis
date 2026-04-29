import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def run_bert_model(sample_csv_path, max_len=64, batch_size=8, epochs=1):
    # Load and prepare dataset
    df = pd.read_csv(sample_csv_path)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})  # binary classification

    X = df['text'].values
    y = df['sentiment'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=max_len)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=max_len)

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(batch_size)

    # Load BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create optimizer
    steps_per_epoch = len(train_dataset)
    num_train_steps = steps_per_epoch * epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps
    )

    # Manual training loop
    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch+1}/{epochs}")
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                inputs, labels = batch
                outputs = model(inputs, training=True)
                logits = outputs.logits
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss.numpy()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"✅ Avg Training Loss: {avg_loss:.4f}")

    # Evaluate
    y_pred_logits = model.predict(test_dataset).logits
    y_pred = tf.argmax(y_pred_logits, axis=1).numpy()

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Save model/tokenizer
    model.save_pretrained('./models/bert_sentiment_model')
    tokenizer.save_pretrained('./models/bert_sentiment_model')

    print("\n✅ BERT model and tokenizer saved successfully!")

    return model, tokenizer
