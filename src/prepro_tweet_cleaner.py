# preprocessing/tweet_cleaner.py
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional

import emoji
import regex  # better unicode support than re
from wordsegment import load as ws_load, segment as ws_segment

# Load wordsegment model once
_ws_loaded = False


@dataclass(frozen=True)
class CleanConfig:
    lowercase: bool = True
    replace_urls: bool = True
    replace_users: bool = True
    demojize: bool = True
    split_hashtags: bool = True
    normalize_elongated: bool = True
    elongated_max_repeats: int = 2  # "soooo" -> "soo"
    keep_punctuation: bool = True  # keep ! ? . , etc.
    strip_rt_prefix: bool = True   # remove leading "RT"
    remove_extra_spaces: bool = True


# --- Precompiled patterns ---
URL_RE = re.compile(r"""(?i)\b(?:https?://|www\.)\S+""")
USER_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
RT_PREFIX_RE = re.compile(r"(?i)^\s*rt\s+@?\w*:?\s*")
HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos);")

# Keep common punctuation; remove other symbols if keep_punctuation=True
# (We also keep angle-bracket tokens like <URL>)
NON_TEXT_RE_KEEP_PUNCT = regex.compile(r"(?!<URL>|<USER>)\p{So}|\p{Cs}", flags=regex.UNICODE)
# Optional: if you want to remove most punctuation, use this later


def _ensure_wordsegment_loaded() -> None:
    global _ws_loaded
    if not _ws_loaded:
        ws_load()
        _ws_loaded = True


def _split_hashtag(token: str) -> str:
    """
    Convert hashtag body into segmented words using wordsegment.
    Example: "HappyDay2024" -> "happy day 2024"
    """
    _ensure_wordsegment_loaded()

    # Separate digits boundaries a bit: HappyDay2024 -> HappyDay 2024
    token = re.sub(r"([A-Za-z])(\d)", r"\1 \2", token)
    token = re.sub(r"(\d)([A-Za-z])", r"\1 \2", token)

    parts = []
    for chunk in token.split():
        if chunk.isdigit():
            parts.append(chunk)
        else:
            parts.extend(ws_segment(chunk))
    return " ".join(parts)


def _normalize_elongated_words(text: str, max_repeats: int = 2) -> str:
    """
    Reduce repeated characters: "sooooo" -> "soo" if max_repeats=2
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeats, text)


def clean_tweet(text: Optional[str], cfg: CleanConfig = CleanConfig()) -> str:
    """
    Clean tweet text for sentiment classification.
    Returns a normalized string (tokens preserved like <URL>, <USER>).
    """
    if text is None:
        return ""

    # 1) HTML unescape (&amp; etc.) + normalize weird entities
    text = html.unescape(text)
    # also handle common &amp; etc if leftover
    text = HTML_ENTITY_RE.sub(" ", text)

    # 2) Remove RT prefix if needed
    if cfg.strip_rt_prefix:
        text = RT_PREFIX_RE.sub("", text)

    # 3) Replace URLs / users
    if cfg.replace_urls:
        text = URL_RE.sub(" <URL> ", text)
    if cfg.replace_users:
        text = USER_RE.sub(" <USER> ", text)

    # 4) Demojize: "😂" -> ":face_with_tears_of_joy:"
    if cfg.demojize:
        text = emoji.demojize(text, delimiters=(" ", " "))

    # 5) Hashtag splitting: "#HappyDay" -> "happy day"
    if cfg.split_hashtags:
        def _hashtag_repl(m: re.Match) -> str:
            body = m.group(1)
            return " " + _split_hashtag(body) + " "
        text = HASHTAG_RE.sub(_hashtag_repl, text)

    # 6) Remove non-text symbols (keep <URL>/<USER>)
    text = NON_TEXT_RE_KEEP_PUNCT.sub(" ", text)

    # 7) Normalize elongated characters
    if cfg.normalize_elongated:
        text = _normalize_elongated_words(text, cfg.elongated_max_repeats)

    # 8) Lowercase
    if cfg.lowercase:
        text = text.lower()

    # 9) Normalize whitespace
    if cfg.remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    samples = [
        "RT @john: soooo happy 😂😂!!! Check https://t.co/abc #HappyDay2024",
        "Ughhh I can't stand this... @support help pls!!! 😡",
        "wowwww this is coooool #MachineLearning #NLP",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", clean_tweet(s))
        print()
