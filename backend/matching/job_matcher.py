"""
Job Matching Engine — embedding similarity + ATS scoring hybrid.
"""
from typing import Any, Dict, List, Optional
from loguru import logger

from backend.ml.embeddings import generate_embedding, cosine_similarity, build_resume_text_for_embedding
from backend.matching.ats_scorer import calculate_ats_score
from backend.nlp.skill_extractor import extract_explicit_skills


def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """Extract structured requirements from raw job description text."""
    from backend.nlp.pipeline import parse_resume  # reuse pipeline
    parsed = parse_resume(jd_text)
    return {
        "required_skills": [
            s["normalized_name"] if isinstance(s, dict) else s
            for s in parsed.get("skills", [])
        ],
        "required_years": parsed.get("total_experience_years", 0),
        "required_degree": (parsed.get("education") or [{}])[0].get("degree"),
        "required_certifications": [c.get("name") for c in parsed.get("certifications", [])],
        "required_languages": [l.get("name") for l in parsed.get("languages", [])],
        "raw_text": jd_text,
    }


def match_candidate_to_job(
    parsed_resume: Dict[str, Any],
    job_requirements: Dict[str, Any],
    resume_embedding: Optional[List[float]] = None,
    job_embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    # ATS score
    ats_result = calculate_ats_score(parsed_resume, job_requirements)

    # Embedding similarity
    sim_score = 0.0
    if resume_embedding and job_embedding:
        sim_score = round(cosine_similarity(resume_embedding, job_embedding) * 100, 2)

    # Hybrid final score: 70% ATS + 30% semantic similarity
    if resume_embedding and job_embedding:
        final = round(0.70 * ats_result["ats_score"] + 0.30 * sim_score, 1)
    else:
        final = ats_result["ats_score"]

    return {
        "final_score":       final,
        "ats_score":         ats_result["ats_score"],
        "semantic_score":    sim_score,
        "breakdown":         ats_result["breakdown"],
        "matched_skills":    ats_result["matched_skills"],
        "missing_skills":    ats_result["missing_skills"],
        "strengths":         ats_result["strengths"],
        "weaknesses":        ats_result["weaknesses"],
        "recommendation":    ats_result["recommendation"],
    }


def rank_candidates(
    candidates: List[Dict[str, Any]],
    job_requirements: Dict[str, Any],
    jd_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rank all candidates against a job description.
    Each candidate dict must have: parsed_resume + optionally embedding vector.
    """
    job_embedding = None
    if jd_text:
        job_embedding = generate_embedding(jd_text)

    results = []
    for candidate in candidates:
        parsed = candidate.get("parsed_resume", {})
        resume_text = build_resume_text_for_embedding(parsed)
        resume_embedding = candidate.get("embedding") or generate_embedding(resume_text)

        match = match_candidate_to_job(parsed, job_requirements, resume_embedding, job_embedding)
        results.append({
            "candidate_id":   candidate.get("id"),
            "full_name":      parsed.get("full_name"),
            "email":          parsed.get("email"),
            "total_exp_years": parsed.get("total_experience_years", 0),
            **match,
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
