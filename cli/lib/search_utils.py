import json
import string
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = str(ROOT_DIR / "cache")
MOVIES_PATH = DATA_DIR / "movies.json"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"
BM25_B = 0.75

_STEMMER = PorterStemmer()
_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def _load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize_text(text: str) -> list[str]:
    normalized = text.lower().translate(_PUNCT_TRANSLATOR)
    stopwords = _load_stopwords()
    tokens = []
    for raw_token in normalized.split():
        if raw_token in stopwords:
            continue
        tokens.append(_STEMMER.stem(raw_token))
    return tokens


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("movies", [])
