"""
PDF metin çıkarıcı.

Strateji:
  1. pdfplumber ile kelime koordinatlarını oku.
  2. Sayfanın iki sütunlu olup olmadığını tespit et.
     - İki sütunlu ise: sol sütunu kendi içinde, sağ sütunu kendi içinde
       yukarıdan aşağıya sırala; sol + sağ olarak birleştir.
     - Tek sütunlu ise: tüm kelimeleri konum sırasıyla oku.
  3. Boşluklu harfleri normalize et (EDUCAT I O N → EDUCATION).
  4. pdfplumber başarısız olursa pdfminer → PyPDF2 fallback zinciri.
"""
import re
from io import BytesIO
from typing import List


# ── Yardımcı: kelime listesini satırlara dönüştür ────────────────────────────

def _words_to_lines(words: list, y_tolerance: float = 4.0) -> str:
    if not words:
        return ""
    lines: List[List[str]] = []
    current_line: List[str] = []
    current_y = words[0]["top"]

    for w in words:
        if abs(w["top"] - current_y) > y_tolerance:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w["text"]]
            current_y = w["top"]
        else:
            current_line.append(w["text"])

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


# ── pdfplumber ile sütun-duyarlı çıkarma ────────────────────────────────────

def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    pages_text: List[str] = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            if not words:
                continue

            page_mid = page.width / 2.0

            left_words  = [w for w in words if w["x0"] < page_mid]
            right_words = [w for w in words if w["x0"] >= page_mid]

            # İki sütunlu mu? Her iki tarafta da anlamlı içerik varsa evet.
            is_two_col = (
                len(left_words) > 5
                and len(right_words) > 5
                and min(len(left_words), len(right_words))
                   / max(len(left_words), len(right_words)) > 0.20
            )

            if is_two_col:
                left_words  = sorted(left_words,  key=lambda w: (w["top"], w["x0"]))
                right_words = sorted(right_words, key=lambda w: (w["top"], w["x0"]))
                left_text   = _words_to_lines(left_words)
                right_text  = _words_to_lines(right_words)
                pages_text.append(left_text + "\n\n" + right_text)
            else:
                all_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
                pages_text.append(_words_to_lines(all_words))

    return "\n\n".join(pages_text).strip()


# ── Fallback: pdfminer ───────────────────────────────────────────────────────

def _extract_with_pdfminer(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        with BytesIO(pdf_bytes) as stream:
            return (extract_text(stream) or "").strip()
    except Exception:
        return ""


# ── Fallback: PyPDF2 ─────────────────────────────────────────────────────────

def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        parts = []
        with BytesIO(pdf_bytes) as stream:
            reader = PdfReader(stream)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception:
        return ""


# ── Metin normalizasyonu ─────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    PDF'den bozuk çıkan spaced-letter satırları düzelt.
    'M a r k e t i n g'  → 'Marketing'
    'EDUCAT I O N'        → 'EDUCATION'
    Sadece satırın büyük çoğunluğu tek karakterden oluşuyorsa işle.
    """
    def fix_line(line: str) -> str:
        tokens = line.strip().split()
        if len(tokens) < 2:
            return line
        single = sum(1 for t in tokens if len(t) == 1)
        if single / len(tokens) >= 0.45:
            return re.sub(r"\s+", "", line.strip())
        return line

    return "\n".join(fix_line(l) for l in text.splitlines())


# ── Ana fonksiyon ────────────────────────────────────────────────────────────

def extract_resume_text(pdf_bytes: bytes) -> str:
    # 1) pdfplumber (en iyi sütun desteği)
    text = _extract_with_pdfplumber(pdf_bytes)

    # 2) pdfminer fallback
    if len(text) < 80:
        text = _extract_with_pdfminer(pdf_bytes)

    # 3) PyPDF2 fallback
    if len(text) < 80:
        text = _extract_with_pypdf2(pdf_bytes)

    return _normalize(text)
