"""
ATS Scoring Engine — weighted multi-factor scoring with explainability.
"""
from typing import Dict, List, Any, Optional
from backend.core.config import get_settings

settings = get_settings()

DEGREE_SCORES = {
    "phd": 1.0, "doctor": 1.0,
    "master": 0.85, "msc": 0.85, "mba": 0.85,
    "bachelor": 0.70, "bsc": 0.70, "b.sc": 0.70,
    "associate": 0.50,
    "high school": 0.30,
}


def _score_skills(candidate_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
    if not required_skills:
        return {"score": 0.80, "matched": [], "missing": [], "ratio": 1.0}
    cand_lower = {s.lower() for s in candidate_skills}
    req_lower  = [s.lower() for s in required_skills]

    matched = [s for s in req_lower if any(s in c or c in s for c in cand_lower)]
    missing = [s for s in required_skills if s.lower() not in {m for m in matched}]
    ratio = len(matched) / len(required_skills) if required_skills else 0
    return {"score": ratio, "matched": matched, "missing": missing, "ratio": ratio}


def _score_experience(total_years: float, required_years: float) -> Dict[str, Any]:
    if required_years <= 0:
        return {"score": 0.80, "candidate_years": total_years, "required_years": 0}
    ratio = min(total_years / required_years, 1.0)
    return {"score": ratio, "candidate_years": total_years, "required_years": required_years}


def _score_education(education: List[Dict], required_degree: Optional[str]) -> Dict[str, Any]:
    if not required_degree or not education:
        return {"score": 0.70, "highest_degree": None}

    req_low = required_degree.lower()
    highest = 0.0
    highest_name = None

    for edu in education:
        deg = (edu.get("degree") or "").lower()
        for keyword, score in DEGREE_SCORES.items():
            if keyword in deg:
                if score > highest:
                    highest = score
                    highest_name = edu.get("degree")

    req_score = next((v for k, v in DEGREE_SCORES.items() if k in req_low), 0.5)
    score = min(highest / req_score, 1.0) if req_score > 0 else 0.70
    return {"score": score, "highest_degree": highest_name}


def _score_certifications(certs: List[Dict], required_certs: List[str]) -> Dict[str, Any]:
    if not required_certs:
        return {"score": 0.80, "matched": []}
    cand_certs = [c.get("name", "").lower() for c in certs]
    matched = [r for r in required_certs if any(r.lower() in cc for cc in cand_certs)]
    ratio = len(matched) / len(required_certs)
    return {"score": ratio, "matched": matched}


def _score_languages(languages: List[Dict], required_langs: List[str]) -> Dict[str, Any]:
    if not required_langs:
        return {"score": 0.80, "matched": []}
    cand_langs = [l.get("name", "").lower() for l in languages]
    matched = [r for r in required_langs if any(r.lower() in cl for cl in cand_langs)]
    ratio = len(matched) / len(required_langs)
    return {"score": ratio, "matched": matched}


def _education_level_score(education: List[Dict]) -> tuple:
    """Highest degree → score, used in standalone profile scoring."""
    highest, highest_name = 0.0, None
    for edu in education:
        deg = (edu.get("degree") or "").lower()
        for keyword, score in DEGREE_SCORES.items():
            if keyword in deg and score > highest:
                highest, highest_name = score, edu.get("degree")
    return highest, highest_name


def calculate_profile_strength(parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone ATS score based on the candidate's intrinsic profile completeness.
    Used when no job description is provided — gives each candidate a distinct score.
    """
    skills = parsed_resume.get("skills", [])
    n_skills = len(skills)
    years = parsed_resume.get("total_experience_years", 0) or 0
    education = parsed_resume.get("education", [])
    certs = parsed_resume.get("certifications", [])
    langs = parsed_resume.get("languages", [])

    # Each component scaled 0..1 based on real content
    skill_score = min(n_skills / 12.0, 1.0)
    exp_score   = min(years / 8.0, 1.0)
    edu_score, edu_name = _education_level_score(education)
    if edu_score == 0.0 and education:
        edu_score = 0.4  # has education but degree not recognized
    cert_score  = min(len(certs) / 3.0, 1.0)
    lang_score  = min(len(langs) / 2.0, 1.0)

    weighted = (
        skill_score * settings.skill_weight +
        exp_score   * settings.experience_weight +
        edu_score   * settings.education_weight +
        cert_score  * settings.certification_weight +
        lang_score  * settings.language_weight
    )
    final = round(weighted * 100, 1)

    strengths, weaknesses = [], []
    if n_skills >= 8:
        strengths.append(f"Broad skill set ({n_skills} skills)")
    elif n_skills < 4:
        weaknesses.append(f"Few skills listed ({n_skills})")
    if years >= 3:
        strengths.append(f"{years} years of experience")
    elif years < 1:
        weaknesses.append("Little or no work experience detected")
    if edu_name:
        strengths.append(f"Education: {edu_name}")
    if certs:
        strengths.append(f"{len(certs)} certification(s)")
    if langs:
        strengths.append(f"{len(langs)} language(s)")

    return {
        "ats_score": final,
        "breakdown": {
            "skills":         {"score": round(skill_score * 100, 1), "weight": f"{int(settings.skill_weight*100)}%"},
            "experience":     {"score": round(exp_score * 100, 1),   "weight": f"{int(settings.experience_weight*100)}%"},
            "education":      {"score": round(edu_score * 100, 1),   "weight": f"{int(settings.education_weight*100)}%"},
            "certifications": {"score": round(cert_score * 100, 1),  "weight": f"{int(settings.certification_weight*100)}%"},
            "languages":      {"score": round(lang_score * 100, 1),  "weight": f"{int(settings.language_weight*100)}%"},
        },
        "matched_skills": [s["normalized_name"] if isinstance(s, dict) else s for s in skills],
        "missing_skills": [],
        "strengths":      strengths,
        "weaknesses":     weaknesses,
        "recommendation": (
            "Strong profile" if final >= 75 else
            "Moderate profile" if final >= 50 else
            "Entry-level profile"
        ),
    }


def calculate_ats_score(
    parsed_resume: Dict[str, Any],
    job_requirements: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns ATS score with full explainability breakdown.

    - If job_requirements is provided → score against the job (matching mode).
    - If not → score the candidate's intrinsic profile strength (standalone mode).
    """
    # Standalone mode: no job → intrinsic profile strength (distinct per candidate)
    if not job_requirements or not any([
        job_requirements.get("required_skills"),
        job_requirements.get("required_years"),
        job_requirements.get("required_degree"),
    ]):
        return calculate_profile_strength(parsed_resume)

    req = job_requirements or {}
    skills_list = [
        s["normalized_name"] if isinstance(s, dict) else s
        for s in parsed_resume.get("skills", [])
    ]

    skill_result = _score_skills(skills_list, req.get("required_skills", []))
    exp_result   = _score_experience(parsed_resume.get("total_experience_years", 0), req.get("required_years", 0))
    edu_result   = _score_education(parsed_resume.get("education", []), req.get("required_degree"))
    cert_result  = _score_certifications(parsed_resume.get("certifications", []), req.get("required_certifications", []))
    lang_result  = _score_languages(parsed_resume.get("languages", []), req.get("required_languages", []))

    weighted = (
        skill_result["score"] * settings.skill_weight +
        exp_result["score"]   * settings.experience_weight +
        edu_result["score"]   * settings.education_weight +
        cert_result["score"]  * settings.certification_weight +
        lang_result["score"]  * settings.language_weight
    )
    final_score = round(weighted * 100, 1)

    # Explainability
    strengths, weaknesses = [], []

    if skill_result["ratio"] >= 0.75:
        strengths.append(f"Strong skill match ({len(skill_result['matched'])} of {len(req.get('required_skills', []))} required skills)")
    elif skill_result["missing"]:
        weaknesses.append(f"Missing skills: {', '.join(skill_result['missing'][:5])}")

    if exp_result["candidate_years"] >= exp_result["required_years"] and exp_result["required_years"] > 0:
        strengths.append(f"Meets experience requirement ({exp_result['candidate_years']} years)")
    elif exp_result["required_years"] > 0:
        weaknesses.append(f"Experience gap ({exp_result['candidate_years']} vs {exp_result['required_years']} required years)")

    if edu_result["highest_degree"]:
        strengths.append(f"Education: {edu_result['highest_degree']}")

    if cert_result["matched"]:
        strengths.append(f"Relevant certifications: {', '.join(cert_result['matched'])}")

    return {
        "ats_score": final_score,
        "breakdown": {
            "skills":         {"score": round(skill_result["score"] * 100, 1), "weight": f"{int(settings.skill_weight*100)}%"},
            "experience":     {"score": round(exp_result["score"] * 100, 1),   "weight": f"{int(settings.experience_weight*100)}%"},
            "education":      {"score": round(edu_result["score"] * 100, 1),   "weight": f"{int(settings.education_weight*100)}%"},
            "certifications": {"score": round(cert_result["score"] * 100, 1),  "weight": f"{int(settings.certification_weight*100)}%"},
            "languages":      {"score": round(lang_result["score"] * 100, 1),  "weight": f"{int(settings.language_weight*100)}%"},
        },
        "matched_skills":  skill_result["matched"],
        "missing_skills":  skill_result["missing"],
        "strengths":       strengths,
        "weaknesses":      weaknesses,
        "recommendation":  (
            "Strong candidate — recommend interview" if final_score >= 75 else
            "Potential candidate — review manually" if final_score >= 50 else
            "Below threshold — consider for junior roles"
        ),
    }
