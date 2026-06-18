"""
Skill normalization — maps variants to canonical form.
Uses exact lookup + RapidFuzz fuzzy matching.
"""
from typing import Optional
from rapidfuzz import process, fuzz

CANONICAL_MAP = {
    # Python variants
    "python 3": "Python", "python3": "Python", "python programming": "Python",
    "python developer": "Python", "py": "Python",
    # JavaScript
    "js": "JavaScript", "javascript es6": "JavaScript", "es6": "JavaScript",
    "ecmascript": "JavaScript", "vanilla js": "JavaScript",
    # TypeScript
    "ts": "TypeScript",
    # Machine Learning
    "ml": "Machine Learning", "machine-learning": "Machine Learning",
    # Deep Learning
    "dl": "Deep Learning", "deep-learning": "Deep Learning",
    # NLP
    "natural language processing": "NLP",
    # REST API
    "rest": "REST API", "restful": "REST API", "restful api": "REST API",
    "rest apis": "REST API", "api development": "REST API",
    # SQL
    "mysql": "MySQL", "ms sql": "MSSQL", "sql server": "MSSQL",
    "postgres": "PostgreSQL", "psql": "PostgreSQL",
    # Cloud
    "amazon web services": "AWS", "google cloud platform": "GCP",
    "microsoft azure": "Azure",
    # DevOps
    "ci cd": "CI/CD", "cicd": "CI/CD", "continuous integration": "CI/CD",
    # Git
    "github": "Git/GitHub", "gitlab": "Git/GitLab", "version control": "Git",
    # React
    "reactjs": "React", "react.js": "React",
    # Node
    "nodejs": "Node.js", "node": "Node.js",
    # Docker/K8s
    "k8s": "Kubernetes", "docker container": "Docker",
    # Soft skills
    "team player": "Teamwork", "team work": "Teamwork",
    "problem-solving": "Problem Solving", "critical-thinking": "Critical Thinking",
}

_CANONICAL_KEYS = list(CANONICAL_MAP.keys())
_CANONICAL_VALUES = sorted(set(CANONICAL_MAP.values()))


def normalize_skill(raw: str, fuzzy_threshold: int = 85) -> str:
    cleaned = raw.strip().lower()

    # 1) Exact lookup
    if cleaned in CANONICAL_MAP:
        return CANONICAL_MAP[cleaned]

    # 2) Fuzzy match against canonical keys
    match = process.extractOne(cleaned, _CANONICAL_KEYS, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= fuzzy_threshold:
        return CANONICAL_MAP[match[0]]

    # 3) Fuzzy match against canonical values (already normalized)
    match2 = process.extractOne(raw, _CANONICAL_VALUES, scorer=fuzz.token_sort_ratio)
    if match2 and match2[1] >= fuzzy_threshold:
        return match2[0]

    # 4) Title-case the original
    return raw.strip().title()


def normalize_skill_list(skills: list) -> list:
    seen, result = set(), []
    for skill in skills:
        norm = normalize_skill(skill["name"] if isinstance(skill, dict) else skill)
        if norm.lower() not in seen:
            seen.add(norm.lower())
            if isinstance(skill, dict):
                result.append({**skill, "normalized_name": norm})
            else:
                result.append(norm)
    return result
