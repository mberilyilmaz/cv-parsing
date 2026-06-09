import re
import string
from typing import Dict, List, Set

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_assets() -> None:
    resource_checks = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    for resource, lookup_path in resource_checks.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(resource, quiet=True)


def get_stopwords() -> Set[str]:
    try:
        ensure_nltk_assets()
        sw = set(stopwords.words("english"))
    except LookupError:
        # NLTK kaynagi indirilemezse uygulama devam etsin.
        sw = set()
    return sw | {
        "ve",
        "ile",
        "icin",
        "için",
        "bir",
        "da",
        "de",
    }


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ").replace("\uf0b7", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> List[str]:
    try:
        ensure_nltk_assets()
        return word_tokenize(text)
    except LookupError:
        # Fallback: regex tabanli tokenization
        return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def remove_stopwords(tokens: List[str]) -> List[str]:
    stopword_set = get_stopwords()
    return [token for token in tokens if token not in stopword_set and len(token) > 1]


def preprocess_resume(text: str) -> Dict[str, object]:
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
    }
