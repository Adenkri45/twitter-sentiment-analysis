# preprocessing.py - Placeholder for Preprocessing logic

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')  # Just in case it wasn't run yet

# You can add more stopwords if needed
STOPWORDS = set(stopwords.words('english'))

def clean_tweet(text):
    """
    Cleans a tweet by removing mentions, hashtags, URLs, punctuation, and stopwords.
    Returns the cleaned, lowercased text.
    """
    text = re.sub(r"@[\w]+", "", text)            # Remove @mentions
    text = re.sub(r"http\S+", "", text)           # Remove URLs
    text = re.sub(r"#", "", text)                 # Remove hashtag symbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # Remove punctuation and numbers
    text = text.lower()                           # Convert to lowercase
    tokens = text.split()                         # Tokenize (split into words)
    tokens = [word for word in tokens if word not in STOPWORDS]  # Remove stopwords
    return " ".join(tokens)
