import re
from typing import List

SKILL_KEYWORDS = {
    # Programlama dilleri
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
    "kotlin", "swift", "r", "scala", "ruby", "php", "dart", "matlab",
    # Web
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
    "django", "flask", "fastapi", "spring", "asp.net",
    # Veri / ML
    "sql", "nosql", "mongodb", "postgresql", "mysql", "sqlite", "redis",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "nlp", "spacy", "nltk", "huggingface", "transformers", "bert", "gpt",
    "computer vision", "opencv", "yolo",
    # DevOps / Cloud
    "docker", "kubernetes", "git", "github", "gitlab", "ci/cd",
    "aws", "azure", "gcp", "google cloud", "heroku", "linux", "bash",
    "terraform", "ansible",
    # Veri mühendisliği
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    "elasticsearch", "tableau", "power bi",
    # Genel
    "streamlit", "jupyter", "excel", "jira", "agile", "scrum",
    "rest", "graphql", "microservices", "oop", "tdd",
    # Soft / yönetim
    "management", "leadership", "creativity", "negotiation",
    "critical thinking", "communication", "teamwork", "problem solving",
    "digital marketing", "marketing", "product management", "project management",
    "time management", "presentation", "analytical",
}

_SEPARATOR_RE = re.compile(r"[,|/;•·]")


def _extract_from_section(section_text: str) -> List[str]:
    """Skill bölümündeki virgülle/maddeyle ayrılmış becerileri toplar."""
    found: List[str] = []
    for line in section_text.splitlines():
        for part in _SEPARATOR_RE.split(line):
            cand = part.strip().lower().strip("•·-– ")
            if 1 < len(cand) <= 40:
                found.append(cand)
    return found


def extract_skills_rule_based(raw_text: str, skills_section: str = "") -> List[str]:
    found: set = set()
    text_l = raw_text.lower()

    # 1) Keyword eşleşmesi (tüm metinde)
    for skill in SKILL_KEYWORDS:
        pattern = re.compile(r"\b" + re.escape(skill) + r"\b")
        if pattern.search(text_l):
            found.add(skill)

    # 2) Skills bölümünden serbest metin
    if skills_section:
        for item in _extract_from_section(skills_section):
            found.add(item)

    # 3) "Skills: ..." satırı varsa genel metinden de al
    for _, raw in re.findall(r"(?:skills?|beceriler?)\s*[:\-]\s*(.+)", text_l):
        for part in _SEPARATOR_RE.split(raw):
            cand = part.strip()
            if 1 < len(cand) <= 40:
                found.add(cand)

    return sorted(found)
