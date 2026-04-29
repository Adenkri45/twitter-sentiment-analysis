import pandas as pd
import os

# File paths
input_path = 'data/sentiment_sample.csv'
output_path = 'data/sentiment_sample_small.csv'

# Only run if output file doesn't already exist
if not os.path.exists(output_path):
    print("📦 Loading sentiment_sample.csv for stratified sampling...")
    
    # Load with warning suppression
    df = pd.read_csv(input_path, encoding='latin-1', header=None, low_memory=False)
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]

    # Filter only desired classes
    df = df[df['sentiment'].isin([0, 2, 4])]

    # Stratified 2% sample per class, with clean index
    sampled_df = df.groupby('sentiment', group_keys=False, observed=True).apply(
        lambda x: x.sample(frac=0.02, random_state=42)
    ).reset_index(drop=True)

    # Save the sampled file
    sampled_df.to_csv(output_path, index=False)
    
    print(f"✅ Stratified sample saved to {output_path} with shape {sampled_df.shape}")
    print(sampled_df['sentiment'].value_counts())
else:
    print("✅ Sample already exists. Skipping creation.")
