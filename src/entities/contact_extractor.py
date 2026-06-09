import re
from typing import Optional


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/\S+", re.IGNORECASE)
GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/\S+", re.IGNORECASE)


def _search(pattern: re.Pattern, text: str) -> Optional[str]:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def extract_contact_fields(text: str) -> dict:
    return {
        "email": _search(EMAIL_RE, text),
        "phone": _search(PHONE_RE, text),
        "linkedin_url": _search(LINKEDIN_RE, text),
        "github_url": _search(GITHUB_RE, text),
    }
