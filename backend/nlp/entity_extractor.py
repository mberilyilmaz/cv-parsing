"""
Named Entity Recognition — spaCy + regex hybrid.
Extracts: PERSON, EMAIL, PHONE, LOCATION, UNIVERSITY, DEGREE,
          COMPANY, JOB_TITLE, SKILL, PROJECT, CERTIFICATION, LANGUAGE
"""
import re
from typing import Dict, List, Optional, Any
from functools import lru_cache
from loguru import logger

EMAIL_RE    = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE    = re.compile(r"\+?[\d][\d\s\-().]{7,}\d")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", re.I)
GITHUB_RE   = re.compile(r"(?:https?://)?(?:www\.)?github\.com/\S+", re.I)

DEGREE_RE = re.compile(
    r"\b(phd|ph\.d|doctor(?:ate)?|master(?:s)?|m\.sc?|msc|mba|bachelor(?:s)?|"
    r"b\.sc?|bsc|b\.a\.|associate|undergraduate|high school|diploma|llb|llm|md)\b",
    re.I,
)

UNIVERSITY_KEYWORDS = [
    "university", "college", "institute", "school of", "academy",
    "polytechnic", "conservatory", "fakulte", "faculty",
]

COMPANY_SUFFIXES = re.compile(
    r"\b(inc\.?|ltd\.?|llc\.?|corp\.?|limited|gmbh|a\.ş\.?|s\.a\.?|"
    r"group|holding|technologies|solutions|systems|consulting|services|agency)\b",
    re.I,
)

JOB_TITLE_RE = re.compile(
    r"\b(engineer|developer|analyst|scientist|architect|designer|manager|"
    r"director|lead|head of|vp|cto|ceo|cfo|consultant|specialist|coordinator|"
    r"officer|intern|trainee|researcher|technician|administrator|supervisor)\b",
    re.I,
)

PROFICIENCY_RE = re.compile(
    r"\b(native|fluent|advanced|intermediate|beginner|basic|professional|elementary)\b",
    re.I,
)


@lru_cache(maxsize=1)
def _load_spacy(model: str = "en_core_web_lg"):
    try:
        import spacy
        return spacy.load(model)
    except OSError:
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("No spaCy model found. NER will use regex only.")
            return None


def _search(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _findall(pattern: re.Pattern, text: str) -> List[str]:
    return [m.strip() for m in pattern.findall(text)]


def extract_contact(text: str) -> Dict[str, Optional[str]]:
    return {
        "email":        _search(EMAIL_RE, text),
        "phone":        _search(PHONE_RE, text),
        "linkedin_url": _search(LINKEDIN_RE, text),
        "github_url":   _search(GITHUB_RE, text),
    }


# Words that disqualify a line from being a name (job titles, section headers, etc.)
_NAME_STOPWORDS = re.compile(
    r"\b(manager|engineer|developer|analyst|scientist|designer|consultant|"
    r"specialist|director|officer|intern|trainee|architect|lead|administrator|"
    r"coordinator|executive|president|founder|student|graduate|resume|cv|"
    r"curriculum|vitae|profile|summary|objective|about|contact|education|"
    r"experience|skills|projects|marketing|sales|finance|software|senior|junior)\b",
    re.I,
)


def _clean_name_candidate(raw: str) -> Optional[str]:
    """Take a raw string, return a clean single-line name or None."""
    # Only the first physical line (spaCy may merge across newlines)
    first = raw.replace("\n", " ").strip()
    first = re.sub(r"\s+", " ", first)
    # Strip trailing job-title fragments
    if _NAME_STOPWORDS.search(first):
        # Cut at the stopword — keep text before it
        first = _NAME_STOPWORDS.split(first)[0].strip()
    words = first.split()
    if not (2 <= len(words) <= 4):
        return None
    if re.search(r"[@:/\d]", first) or len(first) > 50:
        return None
    # Each word should start uppercase (handles ALL CAPS and Title Case)
    if not all(w[0].isupper() for w in words if w):
        return None
    # Normalize ALL-CAPS names to Title Case
    if first.isupper():
        first = first.title()
    return first


def extract_name_spacy(text: str, model: str = "en_core_web_lg") -> Optional[str]:
    # 1) Heuristic on the very first lines is most reliable for resumes
    heuristic = _heuristic_name(text)
    if heuristic:
        return heuristic

    # 2) Fall back to spaCy PERSON entities (cleaned)
    nlp = _load_spacy(model)
    if nlp is not None:
        doc = nlp(text[:2000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned = _clean_name_candidate(ent.text)
                if cleaned:
                    return cleaned
    return None


def _heuristic_name(text: str) -> Optional[str]:
    """Scan the first several non-empty lines for a name-like line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:8]:
        cleaned = _clean_name_candidate(line)
        if cleaned:
            return cleaned
    return None


def extract_entities_spacy(text: str, model: str = "en_core_web_lg") -> Dict[str, List[str]]:
    nlp = _load_spacy(model)
    result: Dict[str, List[str]] = {
        "persons": [], "locations": [], "organizations": [], "dates": []
    }
    if nlp is None:
        return result
    doc = nlp(text[:10000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            result["persons"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            result["locations"].append(ent.text)
        elif ent.label_ == "ORG":
            result["organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            result["dates"].append(ent.text)
    return result


def extract_degree(text: str) -> Optional[str]:
    m = DEGREE_RE.search(text)
    return m.group(0).strip() if m else None


def is_university_line(line: str) -> bool:
    low = line.lower()
    return any(kw in low for kw in UNIVERSITY_KEYWORDS)


def is_company_line(line: str) -> bool:
    return bool(COMPANY_SUFFIXES.search(line))


def is_job_title_line(line: str) -> bool:
    return bool(JOB_TITLE_RE.search(line))


def extract_location_from_text(text: str, model: str = "en_core_web_lg") -> Optional[str]:
    nlp = _load_spacy(model)
    if nlp is None:
        return None
    doc = nlp(text[:3000])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text
    return None


def extract_languages_from_section(section_text: str) -> List[Dict[str, str]]:
    results = []
    for line in section_text.splitlines():
        stripped = line.strip().strip("•·-– ")
        if not stripped or len(stripped) > 80:
            continue
        prof_match = PROFICIENCY_RE.search(stripped)
        lang_name = re.sub(PROFICIENCY_RE, "", stripped).strip().strip("-:·•– ")
        if lang_name:
            results.append({
                "name": lang_name,
                "proficiency": prof_match.group(0).lower() if prof_match else "unknown",
            })
    return results
