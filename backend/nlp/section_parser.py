"""
CV section segmentation — rule-based with fuzzy matching.
Handles spaced-letter headers (EDUCAT I O N → EDUCATION).
"""
import re
from typing import Dict, List

SECTION_MAP: Dict[str, List[str]] = {
    "contact":        ["contact", "personal information", "personal info", "personal details"],
    "summary":        ["summary", "objective", "profile", "about me", "about", "career objective", "professional summary"],
    "education":      ["education", "academic background", "academics", "schooling", "educational background", "academic history"],
    "experience":     ["experience", "work experience", "professional experience", "employment", "work history", "career history", "professional background"],
    "internship":     ["internship", "internships", "trainee", "intern experience"],
    "skills":         ["skills", "technical skills", "core competencies", "competencies", "technologies", "tech stack", "tools & technologies"],
    "projects":       ["projects", "project", "personal projects", "side projects", "academic projects", "portfolio"],
    "certifications": ["certifications", "certification", "certificates", "licenses", "awards", "achievements", "courses", "online courses"],
    "languages":      ["languages", "language skills", "foreign languages"],
    "references":     ["references"],
    "volunteer":      ["volunteer", "volunteering", "community service"],
    "publications":   ["publications", "research", "papers"],
    "hobbies":        ["hobbies", "interests", "extracurricular"],
}

_KW_MAP: Dict[str, str] = {
    kw.replace(" ", ""): label
    for label, kws in SECTION_MAP.items()
    for kw in kws
}


def _normalize_header(line: str) -> str:
    stripped = line.strip().rstrip(":–-— \t")
    tokens = stripped.split()
    if not tokens:
        return ""
    single_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    if len(tokens) >= 2 and single_ratio >= 0.4:
        return re.sub(r"\s+", "", stripped)
    return stripped


def _detect_section(line: str) -> str | None:
    norm = _normalize_header(line)
    if not norm or len(norm) > 60:
        return None
    ns = norm.lower().replace(" ", "")

    # Exact no-space match
    if ns in _KW_MAP:
        return _KW_MAP[ns]

    # Partial: keyword is contained and line isn't much longer
    for kw_ns, label in _KW_MAP.items():
        if kw_ns in ns and len(ns) <= len(kw_ns) + 8:
            return label

    return None


def split_sections(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"

    for line in lines:
        label = _detect_section(line)
        if label:
            current = label
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "".join(v).strip()}
