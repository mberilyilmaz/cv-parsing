"""
Skill extraction — explicit + implicit.
Explicit: keyword matching + section parsing
Implicit: sentence-level inference from job descriptions
"""
import re
from typing import List, Set, Dict

SEPARATOR_RE = re.compile(r"[,|/;•·]")

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go",
        "rust", "kotlin", "swift", "r", "scala", "ruby", "php", "dart", "matlab",
        "perl", "bash", "powershell", "groovy", "lua",
    ],
    "web_frontend": [
        "html", "css", "react", "angular", "vue", "next.js", "nuxt", "svelte",
        "tailwind", "bootstrap", "sass", "webpack", "vite", "jquery",
    ],
    "web_backend": [
        "django", "flask", "fastapi", "spring", "asp.net", "express", "node.js",
        "laravel", "rails", "gin", "fiber", "rest api", "graphql", "grpc",
    ],
    "data_ml": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
        "nlp", "spacy", "nltk", "huggingface", "transformers", "bert", "gpt",
        "computer vision", "opencv", "yolo", "object detection", "image classification",
        "deep learning", "machine learning", "neural network", "reinforcement learning",
        "llm", "rag", "vector database", "langchain",
    ],
    "data_engineering": [
        "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
        "elasticsearch", "redis", "etl", "data pipeline", "data warehouse",
        "databricks", "flink", "hive", "presto",
    ],
    "database": [
        "sql", "postgresql", "mysql", "sqlite", "mongodb", "cassandra", "dynamodb",
        "redis", "neo4j", "oracle", "mssql", "mariadb",
    ],
    "devops_cloud": [
        "docker", "kubernetes", "git", "github", "gitlab", "ci/cd", "jenkins",
        "aws", "azure", "gcp", "google cloud", "heroku", "linux", "terraform",
        "ansible", "helm", "prometheus", "grafana", "nginx", "apache",
    ],
    "analytics": [
        "tableau", "power bi", "looker", "metabase", "excel", "google analytics",
        "a/b testing", "statistical analysis",
    ],
    "soft": [
        "leadership", "teamwork", "communication", "problem solving", "critical thinking",
        "project management", "agile", "scrum", "kanban", "time management",
        "negotiation", "creativity", "analytical", "presentation",
    ],
}

# Flat keyword → category lookup
_ALL_SKILLS: Dict[str, str] = {
    skill: category
    for category, skills in SKILL_TAXONOMY.items()
    for skill in skills
}

# Implicit inference rules: pattern → implied skills
IMPLICIT_RULES: List[tuple] = [
    (re.compile(r"\brest\s*api\b", re.I),           ["REST API", "Backend Development"]),
    (re.compile(r"\bflask\b", re.I),                 ["Python", "Flask", "Backend Development"]),
    (re.compile(r"\bdjango\b", re.I),                ["Python", "Django", "Backend Development"]),
    (re.compile(r"\bfastapi\b", re.I),               ["Python", "FastAPI", "Backend Development"]),
    (re.compile(r"\bpostgresql|postgres\b", re.I),   ["PostgreSQL", "SQL", "Database"]),
    (re.compile(r"\bmongodb\b", re.I),               ["MongoDB", "NoSQL", "Database"]),
    (re.compile(r"\bdocker\b", re.I),                ["Docker", "Containerization", "DevOps"]),
    (re.compile(r"\bkubernetes|k8s\b", re.I),        ["Kubernetes", "Container Orchestration", "DevOps"]),
    (re.compile(r"\breact\b", re.I),                 ["React", "JavaScript", "Frontend Development"]),
    (re.compile(r"\bpytorch\b", re.I),               ["PyTorch", "Python", "Deep Learning", "Machine Learning"]),
    (re.compile(r"\btensorflow\b", re.I),             ["TensorFlow", "Python", "Deep Learning", "Machine Learning"]),
    (re.compile(r"\btransformers|huggingface\b", re.I), ["Transformers", "Python", "NLP", "Machine Learning"]),
    (re.compile(r"\baws\b", re.I),                   ["AWS", "Cloud Computing", "DevOps"]),
    (re.compile(r"\bspark\b", re.I),                 ["Apache Spark", "Big Data", "Scala/Python"]),
    (re.compile(r"\bci/cd|cicd\b", re.I),            ["CI/CD", "DevOps", "Automation"]),
]


def _extract_from_section(section_text: str) -> Set[str]:
    found: Set[str] = set()
    for line in section_text.splitlines():
        for part in SEPARATOR_RE.split(line):
            cand = part.strip().lower().strip("•·-– ")
            if 1 < len(cand) <= 40:
                found.add(cand)
    return found


def extract_explicit_skills(raw_text: str, skills_section: str = "") -> List[Dict]:
    found: Dict[str, str] = {}
    text_l = raw_text.lower()

    for skill, category in _ALL_SKILLS.items():
        pattern = re.compile(r"\b" + re.escape(skill) + r"\b")
        if pattern.search(text_l):
            found[skill] = category

    if skills_section:
        for item in _extract_from_section(skills_section):
            if item not in found:
                matched_cat = next(
                    (cat for sk, cat in _ALL_SKILLS.items() if sk in item or item in sk),
                    "other",
                )
                found[item] = matched_cat

    return [{"name": k, "category": v, "is_implicit": False, "confidence": 1.0}
            for k, v in sorted(found.items())]


def extract_implicit_skills(raw_text: str) -> List[Dict]:
    found: Set[str] = set()
    for pattern, implied in IMPLICIT_RULES:
        if pattern.search(raw_text):
            found.update(implied)
    return [{"name": s, "category": "implicit", "is_implicit": True, "confidence": 0.85}
            for s in sorted(found)]


def extract_all_skills(raw_text: str, skills_section: str = "") -> List[Dict]:
    explicit = extract_explicit_skills(raw_text, skills_section)
    implicit = extract_implicit_skills(raw_text)

    # Merge: don't duplicate explicit skills
    explicit_names = {s["name"] for s in explicit}
    merged = explicit + [s for s in implicit if s["name"].lower() not in explicit_names]
    return merged
