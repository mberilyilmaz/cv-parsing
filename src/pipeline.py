import re
from typing import Optional

from src.entities.contact_extractor import extract_contact_fields
from src.entities.education_extractor import extract_education
from src.entities.experience_extractor import extract_experiences
from src.entities.section_parser import split_into_sections
from src.entities.skill_extractor import extract_skills_rule_based
from src.preprocessing import preprocess_resume


def _extract_name_heuristic(text: str) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if 2 <= len(words) <= 5 and len(stripped) < 70:
            # Tamamen büyük ya da başharfleri büyük isim satırı olabilir
            if not re.search(r"[@:/]", stripped):
                return stripped
    return None


def _extract_languages(section_text: str):
    if not section_text.strip():
        return []
    langs = []
    for line in section_text.splitlines():
        stripped = line.strip().strip("•·-– ")
        if stripped and len(stripped) < 60:
            langs.append(stripped)
    return langs


def _extract_projects(section_text: str):
    if not section_text.strip():
        return []
    projects = []
    current: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                projects.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        projects.append(" ".join(current))
    return projects


def _extract_certifications(section_text: str):
    if not section_text.strip():
        return []
    certs = []
    for line in section_text.splitlines():
        stripped = line.strip().strip("•·-– ")
        if stripped:
            certs.append(stripped)
    return certs


def parse_resume_pipeline(raw_text: str) -> dict:
    sections = split_into_sections(raw_text)
    prep = preprocess_resume(raw_text)
    contact = extract_contact_fields(raw_text)

    # Section bulunamazsa (iki sütunlu PDF vb.) tüm metin üzerinde çalış
    edu_section = sections.get("education", "")
    education   = extract_education(edu_section if edu_section else raw_text)

    _INTERN_RE = re.compile(
        r"\b(intern|internship|staj|stajyer|trainee|apprentice)\b", re.IGNORECASE
    )

    all_exp    = extract_experiences(sections.get("experience", ""))
    internships = extract_experiences(sections.get("internship", ""))

    # experience bölümündeki kayıtları title/company'e göre ayır
    experiences: list = []
    for exp in all_exp:
        title   = exp.get("title") or ""
        company = exp.get("company") or ""
        if _INTERN_RE.search(title) or _INTERN_RE.search(company):
            internships.append(exp)
        else:
            experiences.append(exp)
    skills = extract_skills_rule_based(raw_text, sections.get("skills", ""))
    languages = _extract_languages(sections.get("languages", ""))
    projects = _extract_projects(sections.get("projects", ""))
    certifications = _extract_certifications(sections.get("certifications", ""))

    return {
        "full_name": _extract_name_heuristic(sections.get("header", raw_text)),
        "email": contact["email"],
        "phone": contact["phone"],
        "address": None,
        "linkedin_url": contact["linkedin_url"],
        "github_url": contact["github_url"],
        "education": education,
        "experiences": experiences,
        "internships": internships,
        "skills": skills,
        "technical_skills": skills,
        "projects": projects,
        "certifications": certifications,
        "languages": languages,
        "references": [],
        "cleaned_text": prep["cleaned_text"],
        "tokens": prep["tokens"],
        "filtered_tokens": prep["filtered_tokens"],
    }
