import pytest
from backend.nlp.text_cleaner import clean_resume_text
from backend.nlp.section_parser import split_sections
from backend.nlp.skill_extractor import extract_explicit_skills, extract_implicit_skills
from backend.nlp.skill_normalizer import normalize_skill
from backend.matching.ats_scorer import calculate_ats_score


SAMPLE_RESUME = """
John Doe
john@example.com | +1 555 123 4567 | github.com/johndoe

EDUCATION
New York University
Bachelor of Computer Science
2018 - 2022

EXPERIENCE
Google Inc.
Software Engineer
2022 - 2024
Developed REST APIs using Python and FastAPI. Worked with PostgreSQL and Docker.

SKILLS
Python, FastAPI, PostgreSQL, Docker, React, Machine Learning
"""


def test_clean_text():
    raw = "Hello  \n\n\n World\t!"
    cleaned = clean_resume_text(raw)
    assert "  " not in cleaned


def test_section_parser():
    sections = split_sections(SAMPLE_RESUME)
    assert "education" in sections
    assert "experience" in sections
    assert "skills" in sections


def test_explicit_skill_extraction():
    skills = extract_explicit_skills(SAMPLE_RESUME)
    names = [s["name"] for s in skills]
    assert "python" in names or "Python" in [s["normalized_name"] for s in skills]


def test_implicit_skill_extraction():
    text = "Developed REST APIs using Flask and PostgreSQL."
    skills = extract_implicit_skills(text)
    names = [s["name"] for s in skills]
    assert "Python" in names or "Flask" in names


def test_skill_normalization():
    assert normalize_skill("python 3") == "Python"
    assert normalize_skill("reactjs") == "React"
    assert normalize_skill("k8s") == "Kubernetes"
    assert normalize_skill("postgres") == "PostgreSQL"


def test_ats_scorer_no_requirements():
    parsed = {
        "skills": [{"normalized_name": "Python"}, {"normalized_name": "Docker"}],
        "total_experience_years": 3.0,
        "education": [{"degree": "Bachelor"}],
        "certifications": [],
        "languages": [{"name": "English"}],
    }
    result = calculate_ats_score(parsed)
    assert 0 <= result["ats_score"] <= 100
    assert "breakdown" in result
    assert "strengths" in result


def test_ats_scorer_with_requirements():
    parsed = {
        "skills": [{"normalized_name": "Python"}, {"normalized_name": "Docker"}],
        "total_experience_years": 2.0,
        "education": [{"degree": "Bachelor of Computer Science"}],
        "certifications": [],
        "languages": [{"name": "English"}],
    }
    req = {
        "required_skills": ["Python", "Docker", "Kubernetes"],
        "required_years": 3,
        "required_degree": "Bachelor",
        "required_certifications": [],
        "required_languages": ["English"],
    }
    result = calculate_ats_score(parsed, req)
    assert "missing_skills" in result
    assert "Kubernetes" in result["missing_skills"]
    assert result["ats_score"] < 100
