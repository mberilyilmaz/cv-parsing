from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from loguru import logger

from backend.db.database import get_db
from backend.db.models import Candidate, ResumeEmbedding, Skill, Experience, Education
from backend.matching.job_matcher import parse_job_description, rank_candidates
from backend.matching.ats_scorer import calculate_ats_score
from backend.schemas.resume import JobRequirementSchema, MatchResultSchema

router = APIRouter(prefix="/jobs", tags=["Job Matching"])


@router.post("/job_match", response_model=List[dict])
async def job_match(
    payload: JobRequirementSchema,
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
):
    try:
        # Parse JD to extract requirements
        jd_requirements = parse_job_description(payload.description)
        # Override with explicitly provided fields
        if payload.required_skills:
            jd_requirements["required_skills"] = payload.required_skills
        if payload.required_years:
            jd_requirements["required_years"] = payload.required_years
        if payload.required_degree:
            jd_requirements["required_degree"] = payload.required_degree
        if payload.required_certifications:
            jd_requirements["required_certifications"] = payload.required_certifications
        if payload.required_languages:
            jd_requirements["required_languages"] = payload.required_languages

        # Load candidates with embeddings
        result = await db.execute(
            select(Candidate).join(ResumeEmbedding, isouter=True).limit(500)
        )
        candidates = result.scalars().all()

        # Build candidate dicts for ranking
        candidate_dicts = []
        for c in candidates:
            await db.refresh(c, ["skills", "experiences", "education", "certifications", "languages", "embedding"])
            candidate_dicts.append({
                "id": c.id,
                "embedding": c.embedding.vector if c.embedding else None,
                "parsed_resume": {
                    "full_name": c.full_name,
                    "email": c.email,
                    "total_experience_years": c.total_experience_years,
                    "skills": [{"normalized_name": s.normalized_name, "category": s.category} for s in c.skills],
                    "education": [{"degree": e.degree, "institution": e.institution} for e in c.education],
                    "certifications": [{"name": cert.name} for cert in c.certifications],
                    "languages": [{"name": lang.name} for lang in c.languages],
                },
            })

        ranked = rank_candidates(candidate_dicts, jd_requirements, payload.description)
        return ranked[:limit]

    except Exception as e:
        logger.exception(f"Job matching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/top_candidates", response_model=List[dict])
async def top_candidates(
    payload: JobRequirementSchema,
    db: AsyncSession = Depends(get_db),
    top_n: int = 5,
):
    results = await job_match(payload, db, limit=top_n)
    return results
