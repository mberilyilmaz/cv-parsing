"""
Full NLP pipeline — orchestrates all extraction steps.
"""
import re
from typing import Any, Dict, List, Optional
from loguru import logger

from backend.nlp.text_cleaner import clean_resume_text
from backend.nlp.section_parser import split_sections
from backend.nlp.entity_extractor import (
    extract_contact, extract_name_spacy, extract_degree,
    extract_location_from_text, extract_languages_from_section,
    is_university_line, is_job_title_line, DEGREE_RE,
)
from backend.nlp.skill_extractor import extract_all_skills
from backend.nlp.skill_normalizer import normalize_skill_list
from backend.core.config import get_settings

settings = get_settings()

YEAR_RANGE_RE = re.compile(
    r"\b(19|20)\d{2}\s*[-–—]\s*(?:(19|20)\d{2}|present|current|now)\b", re.I
)
SINGLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
GPA_RE = re.compile(r"\b(?:gpa|cgpa|grade)[:\s]*([0-9]+[.,][0-9]+)\b", re.I)
INTERN_RE = re.compile(r"\b(intern|internship|trainee|apprentice)\b", re.I)
MONTH_YEAR_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(19|20)\d{2}\b", re.I
)


def _split_blocks(text: str) -> List[List[str]]:
    blocks, current = [], []
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


def _parse_period(text: str) -> Optional[str]:
    m = YEAR_RANGE_RE.search(text)
    if m:
        return m.group(0)
    m2 = MONTH_YEAR_RE.search(text)
    if m2:
        return m2.group(0)
    m3 = SINGLE_YEAR_RE.search(text)
    return m3.group(0) if m3 else None


def _estimate_duration_months(period: Optional[str]) -> int:
    if not period:
        return 0
    years = SINGLE_YEAR_RE.findall(period)
    if len(years) >= 2:
        try:
            return (int(years[1]) - int(years[0])) * 12
        except Exception:
            return 0
    return 6  # rough estimate for single year


def _parse_education_blocks(section_text: str) -> List[Dict]:
    results = []
    for block in _split_blocks(section_text):
        text = "\n".join(block)
        institution = next((l for l in block if is_university_line(l)), None)
        degree_m = DEGREE_RE.search(text)
        degree = degree_m.group(0).strip() if degree_m else None
        period = _parse_period(text)
        gpa_m = GPA_RE.search(text)
        gpa = gpa_m.group(1) if gpa_m else None

        used = {institution, degree}
        field = None
        for line in block:
            if line in used:
                continue
            if YEAR_RANGE_RE.search(line) and len(line.split()) <= 5:
                continue
            field = line
            break

        if institution or degree:
            results.append({
                "institution": institution,
                "degree": degree,
                "field_of_study": field,
                "period": period,
                "gpa": gpa,
            })
    return results


def _parse_experience_blocks(section_text: str, entry_type: str = "work") -> List[Dict]:
    results = []
    for block in _split_blocks(section_text):
        text = "\n".join(block)
        period = _parse_period(text)
        is_current = bool(re.search(r"\b(present|current|now)\b", text, re.I))
        duration = _estimate_duration_months(period)

        title_line = next((l for l in block if is_job_title_line(l)), None)
        from backend.nlp.entity_extractor import is_company_line, COMPANY_SUFFIXES
        company_line = next(
            (l for l in block if is_company_line(l) and l != title_line), None
        )

        if not title_line and not company_line:
            title_line = block[0] if block else None
            company_line = block[1] if len(block) > 1 else None
        elif title_line and not company_line:
            company_line = next(
                (l for l in block if l != title_line and not YEAR_RANGE_RE.search(l)), None
            )
        elif company_line and not title_line:
            title_line = next(
                (l for l in block if l != company_line and not YEAR_RANGE_RE.search(l)), None
            )

        used = {title_line, company_line}
        desc_lines = [
            l for l in block
            if l not in used and not (YEAR_RANGE_RE.search(l) and len(l.split()) <= 5)
        ]
        description = " ".join(desc_lines) if desc_lines else None

        t = entry_type
        if INTERN_RE.search(title_line or "") or INTERN_RE.search(company_line or ""):
            t = "internship"

        if company_line or title_line:
            results.append({
                "company":        company_line,
                "job_title":      title_line,
                "period":         period,
                "is_current":     is_current,
                "duration_months": duration,
                "description":    description,
                "entry_type":     t,
            })
    return results


def _parse_projects(section_text: str) -> List[Dict]:
    results = []
    for block in _split_blocks(section_text):
        name = block[0] if block else None
        desc = " ".join(block[1:]) if len(block) > 1 else None
        url_m = re.search(r"https?://\S+", " ".join(block))
        techs = [s for s in re.findall(r"\b\w[\w.+#-]{1,20}\b", " ".join(block))
                 if s.lower() in {sk: True for sk in []}]  # placeholder
        results.append({"name": name, "description": desc, "technologies": [], "url": url_m.group(0) if url_m else None})
    return results


def _parse_certifications(section_text: str) -> List[Dict]:
    results = []
    for line in section_text.splitlines():
        stripped = line.strip().strip("•·-– ")
        if stripped and len(stripped) > 3:
            year_m = SINGLE_YEAR_RE.search(stripped)
            name = re.sub(SINGLE_YEAR_RE, "", stripped).strip(" -–")
            results.append({"name": name, "issuer": None, "date": year_m.group(0) if year_m else None})
    return results


def _total_experience_years(experiences: List[Dict]) -> float:
    total = sum(e.get("duration_months", 0) for e in experiences if e.get("entry_type") == "work")
    return round(total / 12, 1)


def parse_resume(raw_text: str) -> Dict[str, Any]:
    logger.info("Starting NLP pipeline...")
    cleaned = clean_resume_text(raw_text)
    sections = split_sections(cleaned)

    contact = extract_contact(cleaned)
    name = extract_name_spacy(sections.get("header", cleaned), settings.spacy_model)
    location = extract_location_from_text(sections.get("header", cleaned)[:2000], settings.spacy_model)

    edu_text = sections.get("education", "")
    education = _parse_education_blocks(edu_text if edu_text else cleaned)

    exp_text = sections.get("experience", "")
    int_text = sections.get("internship", "")
    experiences = _parse_experience_blocks(exp_text)
    experiences += _parse_experience_blocks(int_text, entry_type="internship")

    raw_skills = extract_all_skills(cleaned, sections.get("skills", ""))
    skills = normalize_skill_list(raw_skills)

    languages = extract_languages_from_section(sections.get("languages", ""))
    projects = _parse_projects(sections.get("projects", ""))
    certifications = _parse_certifications(sections.get("certifications", ""))

    total_years = _total_experience_years(experiences)

    summary_section = sections.get("summary", "")

    logger.info(f"Pipeline complete: {len(skills)} skills, {len(education)} edu, {len(experiences)} exp")

    return {
        "full_name":               name,
        "email":                   contact["email"],
        "phone":                   contact["phone"],
        "location":                location,
        "linkedin_url":            contact["linkedin_url"],
        "github_url":              contact["github_url"],
        "summary":                 summary_section or None,
        "education":               education,
        "experiences":             experiences,
        "skills":                  skills,
        "languages":               languages,
        "projects":                projects,
        "certifications":          certifications,
        "total_experience_years":  total_years,
        "cleaned_text":            cleaned,
        "sections_detected":       list(sections.keys()),
    }
