"""
Multi-layer OCR extraction pipeline.
Priority: pdfplumber → PyMuPDF → EasyOCR (for scanned/image PDFs)
"""
import io
import re
from pathlib import Path
from typing import Optional
from loguru import logger


def _normalize_spaced_text(text: str) -> str:
    def fix_line(line: str) -> str:
        tokens = line.strip().split()
        if len(tokens) >= 2 and sum(1 for t in tokens if len(t) == 1) / len(tokens) >= 0.45:
            return re.sub(r"\s+", "", line.strip())
        return line
    return "\n".join(fix_line(l) for l in text.splitlines())


def _words_to_lines(words: list, y_tol: float = 4.0) -> str:
    if not words:
        return ""
    lines, current, cy = [], [], words[0]["top"]
    for w in words:
        if abs(w["top"] - cy) > y_tol:
            if current:
                lines.append(" ".join(current))
            current, cy = [w["text"]], w["top"]
        else:
            current.append(w["text"])
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def extract_with_pdfplumber(data: bytes) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=False)
                if not words:
                    continue
                mid = page.width / 2.0
                left  = sorted([w for w in words if w["x0"] < mid],  key=lambda w: (w["top"], w["x0"]))
                right = sorted([w for w in words if w["x0"] >= mid], key=lambda w: (w["top"], w["x0"]))
                is_two_col = (
                    len(left) > 5 and len(right) > 5
                    and min(len(left), len(right)) / max(len(left), len(right)) > 0.20
                )
                if is_two_col:
                    pages.append(_words_to_lines(left) + "\n\n" + _words_to_lines(right))
                else:
                    pages.append(_words_to_lines(sorted(words, key=lambda w: (w["top"], w["x0"]))))
        return "\n\n".join(pages).strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        return ""


def extract_with_pymupdf(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages).strip()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
        return ""


def extract_with_easyocr(data: bytes) -> str:
    try:
        import fitz
        import easyocr
        import numpy as np

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        doc = fitz.open(stream=data, filetype="pdf")
        all_text = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            results = reader.readtext(img_array, detail=0, paragraph=True)
            all_text.append("\n".join(results))
        doc.close()
        return "\n\n".join(all_text).strip()
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
        return ""


def extract_with_image_easyocr(data: bytes) -> str:
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        return "\n".join(results).strip()
    except Exception as e:
        logger.warning(f"Image EasyOCR failed: {e}")
        return ""


def extract_resume_text(data: bytes, filename: str = "") -> str:
    """
    Main extraction entry point.
    Supports: PDF, PNG, JPG, JPEG
    """
    ext = Path(filename).suffix.lower() if filename else ".pdf"

    if ext in {".png", ".jpg", ".jpeg"}:
        logger.info("Image file detected — using EasyOCR")
        text = extract_with_image_easyocr(data)
        return _normalize_spaced_text(text)

    # PDF pipeline
    logger.info("Trying pdfplumber...")
    text = extract_with_pdfplumber(data)

    if len(text.strip()) < 80:
        logger.info("pdfplumber insufficient — trying PyMuPDF...")
        text = extract_with_pymupdf(data)

    if len(text.strip()) < 80:
        logger.info("PyMuPDF insufficient — falling back to EasyOCR (scanned PDF)...")
        text = extract_with_easyocr(data)

    return _normalize_spaced_text(text)
