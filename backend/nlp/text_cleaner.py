import re
import unicodedata
from typing import List


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def fix_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " URL ", text)


def clean_resume_text(raw: str) -> str:
    text = normalize_unicode(raw)
    text = remove_control_chars(text)
    text = fix_whitespace(text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())
