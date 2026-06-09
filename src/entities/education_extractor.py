"""
Eğitim bölümünden okul, bölüm, derece ve yıl bilgilerini çeker.

Strateji:
  - Metni boş satırlara göre bloklara böl
  - Her blokta: kurum satırı (university/lise keyword), derece satırı,
    alan satırı ve yıl satırını tespit et
  - Kurum keyword'ü içeren satır bloğun herhangi bir yerinde olabilir
    (degree-first layout da desteklenir)
"""
import re
from typing import Any, Dict, List

INSTITUTION_KEYWORDS = [
    "üniversite", "university", "universite",
    "college", "okul", "school",
    "lise", "high school", "anadolu",
    "enstitü", "institute", "institut",
    "akademi", "academy",
    "polytechnic", "conservatory",
]

DEGREE_KEYWORDS = [
    "phd", "ph.d", "doktora",
    "master", "masters", "m.sc", "msc", "m.s.", "yüksek lisans", "yuksek lisans",
    "bachelor", "bachelors", "b.sc", "bsc", "b.s.", "lisans", "undergraduate",
    "associate", "önlisans", "onlisans",
    "mba", "llb", "llm", "md",
]

DEGREE_RE = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in DEGREE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
YEAR_RANGE_RE = re.compile(
    r"\b(19|20)\d{2}\s*[-–—]\s*(?:(19|20)\d{2}|present|günümüz|devam|halen|current)\b",
    re.IGNORECASE,
)
SINGLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
GPA_RE = re.compile(
    r"\b(?:gpa|cgpa|not ortalaması|ortalama)[:\s]*([0-9]+[.,][0-9]+)\b",
    re.IGNORECASE,
)


def _is_institution(line: str) -> bool:
    low = line.lower()
    return any(kw in low for kw in INSTITUTION_KEYWORDS)


def _is_date_only(line: str) -> bool:
    stripped = line.strip()
    return bool(YEAR_RANGE_RE.fullmatch(stripped) or
                re.fullmatch(r"[\d\s\-–—/.]+", stripped))


def _split_blocks(text: str) -> List[List[str]]:
    """Boş satırlara göre metin bloklarına böl."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks


def _parse_block(lines: List[str]) -> Dict[str, Any]:
    block_text = "\n".join(lines)

    # Kurum
    institution = next((l for l in lines if _is_institution(l)), None)

    # Derece — önce tam satır eşleşmesi, sonra parçalı
    degree_line = next((l for l in lines if DEGREE_RE.search(l)), None)
    degree = DEGREE_RE.search(degree_line).group(0) if degree_line else None

    # Dönem
    yr_m = YEAR_RANGE_RE.search(block_text)
    period = yr_m.group(0) if yr_m else None
    if not period:
        sy_m = SINGLE_YEAR_RE.search(block_text)
        period = sy_m.group(0) if sy_m else None

    # GPA
    gpa_m = GPA_RE.search(block_text)
    gpa = gpa_m.group(1) if gpa_m else None

    # Alan / bölüm: kurum, derece satırı ve saf tarih satırı dışındaki ilk anlamlı satır
    used = {institution, degree_line}
    field_of_study = None
    for line in lines:
        if line in used:
            continue
        if _is_date_only(line):
            continue
        if YEAR_RANGE_RE.search(line) and len(line.split()) <= 5:
            continue
        field_of_study = line
        break

    return {
        "institution": institution,
        "degree": degree,
        "field_of_study": field_of_study,
        "period": period,
        "gpa": gpa,
    }


def extract_education(section_text: str) -> List[Dict[str, Any]]:
    if not section_text.strip():
        return []

    blocks = _split_blocks(section_text)

    # Bloklar yoksa ya da hiçbirinde kurum/derece yok → tek büyük blok dene
    if not blocks:
        return []

    results: List[Dict[str, Any]] = []
    for block in blocks:
        rec = _parse_block(block)
        # En az kurum veya derece bilgisi olan kayıtları al
        if rec["institution"] or rec["degree"]:
            results.append(rec)

    return results
