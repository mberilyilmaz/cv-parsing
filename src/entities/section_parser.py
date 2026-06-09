"""
CV metnini bölümlere ayırır.
Başlık tespiti: satırın tamamı bir section keyword içeriyorsa (exact veya partial).
"""
import re
from typing import Dict, List

SECTION_MAP: Dict[str, List[str]] = {
    "contact":        ["contact", "iletişim", "kişisel bilgiler", "personal info",
                       "personal information", "kisisel bilgiler"],
    "summary":        ["summary", "objective", "profile", "about me", "about",
                       "özet", "ozet", "hakkımda", "hakkimda", "kariyer hedefi"],
    "education":      ["education", "eğitim", "egitim", "academic background",
                       "academics", "schooling", "eğitim bilgileri", "egitim bilgileri",
                       "educational background", "eğitim durumu", "egitim durumu"],
    "experience":     ["experience", "work experience", "iş deneyimi", "is deneyimi",
                       "professional experience", "employment", "work history",
                       "kariyer", "deneyim", "çalışma deneyimi", "calisma deneyimi",
                       "professional background", "is gecmisi", "iş geçmişi"],
    "internship":     ["internship", "internships", "staj", "stajlar",
                       "staj deneyimi", "trainee", "intern experience"],
    "skills":         ["skills", "technical skills", "core competencies",
                       "beceriler", "yetenekler", "teknik beceriler",
                       "competencies", "technologies", "teknolojiler",
                       "programlama", "araçlar", "araclar", "tools"],
    "projects":       ["projects", "project", "projeler", "personal projects",
                       "side projects", "akademik projeler", "proje"],
    "certifications": ["certifications", "certification", "certificates",
                       "sertifikalar", "sertifika", "licenses", "awards",
                       "achievements", "başarılar", "basarilar",
                       "online kurslar", "kurslar", "courses"],
    "languages":      ["languages", "diller", "language skills", "yabancı dil",
                       "yabanci dil", "foreign languages"],
    "references":     ["references", "referanslar", "referans"],
    "volunteer":      ["volunteer", "volunteering", "gönüllülük", "gonulluluk",
                       "toplum hizmetleri"],
    "hobbies":        ["hobbies", "interests", "hobiler", "ilgi alanları",
                       "ilgi alanlari", "extracurricular"],
}

# Tüm anahtar kelimeleri küçük harfli set olarak hazırla
_ALL_KEYWORDS: Dict[str, str] = {}  # keyword -> label
for _label, _keywords in SECTION_MAP.items():
    for _kw in _keywords:
        _ALL_KEYWORDS[_kw.lower()] = _label


def _normalize_header(line: str) -> str:
    """
    PDF'den bozuk çıkan başlıkları normalize eder.
    'EDUCAT I O N' → 'EDUCATION'
    'AB O UT ME'   → 'ABOUTME'  → eşleşme için boşluksuz karşılaştırma yapılır.
    """
    stripped = line.strip().rstrip(":–-— \t")
    tokens = stripped.split()
    if not tokens:
        return ""
    single = sum(1 for t in tokens if len(t) == 1)
    # Token'ların yarısından fazlası tek karakter → spaced başlık
    if len(tokens) >= 2 and single / len(tokens) >= 0.4:
        return re.sub(r"\s+", "", stripped)
    return stripped


def _detect_section(line: str) -> str | None:
    """
    Satırın bir section başlığı olup olmadığını tespit eder.
    Normal, ALL-CAPS ve boşluklu (spaced-letter) başlıkları tanır.
    """
    normalized = _normalize_header(line)
    if not normalized or len(normalized) > 60:
        return None

    norm = normalized.lower()

    # 1) Boşluklar kaldırılmış karşılaştırma (hem keyword hem satır)
    norm_ns = norm.replace(" ", "")
    for kw, label in _ALL_KEYWORDS.items():
        kw_ns = kw.replace(" ", "")
        if norm_ns == kw_ns:
            return label
        # Satır keyword'ü tam kapsıyor ve fazladan çok şey yok
        if kw_ns in norm_ns and len(norm_ns) <= len(kw_ns) + 6:
            return label

    # 2) Normal partial match
    for kw, label in _ALL_KEYWORDS.items():
        if kw in norm and len(norm) <= len(kw) + 15:
            return label

    return None


def split_into_sections(text: str) -> Dict[str, str]:
    """
    Döndürür: {"education": "...", "skills": "...", ...}
    CV başı (isim, iletişim) "header" anahtarına atanır.
    """
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"

    for line in lines:
        label = _detect_section(line)
        if label:
            current = label
            if current not in sections:
                sections[current] = []
        else:
            if current not in sections:
                sections[current] = []
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}
