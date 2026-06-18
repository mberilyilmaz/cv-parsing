from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import csv, io
from fastapi.responses import StreamingResponse

from backend.db.database import get_db
from backend.db.models import Candidate, Skill

router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.get("/candidate_search", response_model=List[dict])
async def candidate_search(
    skill: Optional[str] = Query(None),
    min_experience: float = Query(0),
    max_experience: float = Query(50),
    degree: Optional[str] = Query(None),
    min_ats: float = Query(0),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    query = select(Candidate).where(
        Candidate.total_experience_years >= min_experience,
        Candidate.total_experience_years <= max_experience,
        Candidate.ats_score >= min_ats,
    )

    if skill:
        skill_subq = select(Skill.candidate_id).where(
            Skill.normalized_name.ilike(f"%{skill}%")
        )
        query = query.where(Candidate.id.in_(skill_subq))

    query = query.order_by(Candidate.ats_score.desc()).limit(limit)
    result = await db.execute(query)
    candidates = result.scalars().all()

    return [
        {
            "id": c.id,
            "full_name": c.full_name,
            "email": c.email,
            "location": c.location,
            "total_experience_years": c.total_experience_years,
            "ats_score": c.ats_score,
            "created_at": c.created_at.isoformat(),
        }
        for c in candidates
    ]


@router.get("/{candidate_id}", response_model=dict)
async def get_candidate(candidate_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    await db.refresh(candidate, ["skills", "experiences", "education", "projects", "certifications", "languages"])
    return {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "email": candidate.email,
        "phone": candidate.phone,
        "location": candidate.location,
        "linkedin_url": candidate.linkedin_url,
        "github_url": candidate.github_url,
        "summary": candidate.summary,
        "total_experience_years": candidate.total_experience_years,
        "ats_score": candidate.ats_score,
        "skills": [{"name": s.normalized_name, "category": s.category} for s in candidate.skills],
        "education": [{"institution": e.institution, "degree": e.degree, "field": e.field_of_study} for e in candidate.education],
        "experiences": [{"company": e.company, "title": e.job_title, "period": e.period} for e in candidate.experiences],
        "certifications": [{"name": c.name} for c in candidate.certifications],
        "languages": [{"name": l.name, "proficiency": l.proficiency} for l in candidate.languages],
    }


@router.delete("/{candidate_id}", response_model=dict)
async def delete_candidate(candidate_id: int, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete as sa_delete
    from backend.db.models import Skill, Education, Experience, Project, Certification, Language, ResumeEmbedding

    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    for model in (Skill, Education, Experience, Project, Certification, Language, ResumeEmbedding):
        await db.execute(sa_delete(model).where(model.candidate_id == candidate_id))
    await db.execute(sa_delete(Candidate).where(Candidate.id == candidate_id))
    await db.commit()
    return {"deleted": candidate_id, "message": "Candidate deleted successfully"}


@router.get("/export/csv")
async def export_candidates_csv(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).order_by(Candidate.ats_score.desc()).limit(1000))
    candidates = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Name", "Email", "Phone", "Location", "Experience Years", "ATS Score", "Created At"])
    for c in candidates:
        writer.writerow([c.id, c.full_name, c.email, c.phone, c.location,
                         c.total_experience_years, c.ats_score, c.created_at])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=candidates.csv"},
    )
