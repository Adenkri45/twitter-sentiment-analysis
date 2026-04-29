from src.prepro_tweet_cleaner import clean_tweet


samples = [
    "RT @john: soooo happy 😂!!! https://t.co/abc #HappyDay2024",
    "Ughhh I can't stand this... @support help pls!!! 😡",
    "wowwww this is coooool #MachineLearning #NLP",
]

for s in samples:
    print("IN :", s)
    print("OUT:", clean_tweet(s))
    print("-" * 40)
