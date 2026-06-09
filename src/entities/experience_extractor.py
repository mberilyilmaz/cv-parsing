"""
İş deneyimi ve staj bölümlerinden pozisyon, şirket ve tarih bilgilerini çeker.

Strateji:
  - Her boş-satır-ayrımlı blok = bir deneyim kaydı
  - Blok içinde title satırı (unvan keyword içeren) ve company satırı ayrıştırılır
  - Title önce gelse de sonra gelse de doğru atanır
"""
import re
from typing import Any, Dict, List

YEAR_RANGE_RE = re.compile(
    r"\b(19|20)\d{2}\s*[-–—]\s*(?:(19|20)\d{2}|present|günümüz|devam|halen|current)\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    r"|oca|şub|mar|nis|may|haz|tem|ağu|eyl|eki|kas|ara)"
    r"[a-zğüşıöç]*\.?\s*(19|20)\d{2}\b",
    re.IGNORECASE,
)

# Unvan içeren satırı tespit etmek için
TITLE_RE = re.compile(
    r"\b(engineer|developer|analyst|intern|stajyer|manager|lead|architect|"
    r"scientist|designer|consultant|specialist|coordinator|director|officer|"
    r"mühendis|geliştirici|uzman|danışman|müdür|asistan|assistant|"
    r"researcher|technician|supervisor|executive|head of|vp|cto|ceo|cfo)\b",
    re.IGNORECASE,
)

# Şirket adı olabilecek satırı tespit etmek için (Ltd, Inc, A.Ş. vb.)
COMPANY_RE = re.compile(
    r"\b(ltd|limited|inc|corp|corporation|llc|gmbh|a\.ş|aş|şirketi|"
    r"company|co\.|group|holding|teknoloji|technology|solutions|services|"
    r"industries|systems|consulting|agency)\b",
    re.IGNORECASE,
)


def _split_entries(section_text: str) -> List[List[str]]:
    """Boş satırlara göre bloklara böl, her blok satır listesi."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        blocks.append(current)
    return blocks


def _parse_block(lines: List[str]) -> Dict[str, Any]:
    block_text = "\n".join(lines)

    # Dönem
    yr = YEAR_RANGE_RE.search(block_text)
    period = yr.group(0) if yr else None
    if not period:
        mo = MONTH_YEAR_RE.search(block_text)
        period = mo.group(0) if mo else None

    # Title satırını bul: TITLE_RE eşleşen ilk satır
    title_line    = next((l for l in lines if TITLE_RE.search(l)), None)
    # Company satırını bul: COMPANY_RE eşleşen satır veya title olmayan ilk kısa satır
    company_line  = next((l for l in lines if COMPANY_RE.search(l) and l != title_line), None)

    # Hiçbir keyword eşleşmezse: title birinci satır, company ikinci satır (eski davranış)
    if not title_line and not company_line:
        title_line   = lines[0] if lines else None
        company_line = lines[1] if len(lines) > 1 else None
    elif title_line and not company_line:
        # Title bulundu ama company yok → diğer kısa satırdan tahmin et
        for l in lines:
            if l != title_line and not YEAR_RANGE_RE.search(l) and not MONTH_YEAR_RE.search(l):
                company_line = l
                break
    elif company_line and not title_line:
        # Company bulundu ama title yok
        for l in lines:
            if l != company_line and not YEAR_RANGE_RE.search(l) and not MONTH_YEAR_RE.search(l):
                title_line = l
                break

    # Açıklama: title, company ve saf tarih satırları dışındaki satırlar
    used = {title_line, company_line}
    desc_lines = []
    for l in lines:
        if l in used:
            continue
        if YEAR_RANGE_RE.search(l) and len(l.split()) <= 5:
            continue
        if MONTH_YEAR_RE.search(l) and len(l.split()) <= 5:
            continue
        desc_lines.append(l)
    description = " - ".join(desc_lines) if desc_lines else None

    return {
        "company": company_line,
        "title":   title_line,
        "period":  period,
        "description": description,
    }


def extract_experiences(section_text: str) -> List[Dict[str, Any]]:
    if not section_text.strip():
        return []
    results = []
    for block in _split_entries(section_text):
        rec = _parse_block(block)
        if rec["company"] or rec["title"]:
            results.append(rec)
    return results
