from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from loguru import logger

from backend.db.database import get_db
from backend.db.models import Candidate, Skill, Education, Experience, Project, Certification, Language, ResumeEmbedding
from backend.ocr.extractor import extract_resume_text
from backend.nlp.pipeline import parse_resume
from backend.matching.ats_scorer import calculate_ats_score
from backend.ml.embeddings import generate_embedding, build_resume_text_for_embedding
from backend.schemas.resume import ParsedResumeSchema, CandidateResponse

router = APIRouter(prefix="/resume", tags=["Resume"])

ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/jpg"}


def _name_from_filename(filename: str) -> str | None:
    """Derive a clean person name from a filename like 'Lorna Alvarado.pdf'."""
    import re
    from pathlib import Path
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem)              # underscores/dashes → space
    stem = re.sub(r"\b(cv|resume|resmi|ozgecmis)\b", "", stem, flags=re.I)
    stem = re.sub(r"\s+", " ", stem).strip()
    words = stem.split()
    if 2 <= len(words) <= 3 and all(w.isalpha() for w in words):
        return stem.title()
    return None


def _refine_name(parsed_name: str | None, filename: str) -> str | None:
    """Prefer a clean filename-derived name when parsing is missing or unreliable."""
    fname = _name_from_filename(filename or "")
    if not fname:
        return parsed_name
    if not parsed_name:
        return fname
    # If parsed name shares any token with the filename name, trust the filename
    parsed_tokens = {w.lower() for w in parsed_name.split()}
    fname_tokens = {w.lower() for w in fname.split()}
    if parsed_tokens & fname_tokens:
        return fname
    return parsed_name


async def _replace_existing_by_email(db: AsyncSession, email: str | None) -> None:
    """If a candidate with this email exists, delete it so re-upload replaces it."""
    if not email:
        return
    from sqlalchemy import select, delete as sa_delete
    existing = await db.execute(select(Candidate).where(Candidate.email == email))
    old = existing.scalar_one_or_none()
    if old:
        for model in (Skill, Education, Experience, Project, Certification, Language, ResumeEmbedding):
            await db.execute(sa_delete(model).where(model.candidate_id == old.id))
        await db.execute(sa_delete(Candidate).where(Candidate.id == old.id))
        await db.flush()


async def _save_candidate(db: AsyncSession, parsed: dict, raw_text: str, ats: dict) -> Candidate:
    await _replace_existing_by_email(db, parsed.get("email"))
    candidate = Candidate(
        full_name=parsed.get("full_name"),
        email=parsed.get("email"),
        phone=parsed.get("phone"),
        location=parsed.get("location"),
        linkedin_url=parsed.get("linkedin_url"),
        github_url=parsed.get("github_url"),
        summary=parsed.get("summary"),
        raw_text=raw_text,
        cleaned_text=parsed.get("cleaned_text"),
        total_experience_years=parsed.get("total_experience_years", 0),
        ats_score=ats.get("ats_score", 0),
    )
    db.add(candidate)
    await db.flush()

    for s in parsed.get("skills", []):
        db.add(Skill(
            candidate_id=candidate.id,
            raw_name=s.get("name", s) if isinstance(s, dict) else s,
            normalized_name=s.get("normalized_name", s.get("name", s)) if isinstance(s, dict) else s,
            category=s.get("category") if isinstance(s, dict) else None,
            is_implicit=s.get("is_implicit", False) if isinstance(s, dict) else False,
            confidence=s.get("confidence", 1.0) if isinstance(s, dict) else 1.0,
        ))

    for e in parsed.get("education", []):
        db.add(Education(
            candidate_id=candidate.id,
            institution=e.get("institution"),
            degree=e.get("degree"),
            field_of_study=e.get("field_of_study"),
            period=e.get("period"),
            gpa=e.get("gpa"),
        ))

    for exp in parsed.get("experiences", []):
        db.add(Experience(
            candidate_id=candidate.id,
            company=exp.get("company"),
            job_title=exp.get("job_title"),
            period=exp.get("period"),
            duration_months=exp.get("duration_months"),
            description=exp.get("description"),
            is_current=exp.get("is_current", False),
            entry_type=exp.get("entry_type", "work"),
        ))

    for proj in parsed.get("projects", []):
        db.add(Project(
            candidate_id=candidate.id,
            name=proj.get("name"),
            description=proj.get("description"),
            technologies=proj.get("technologies", []),
            url=proj.get("url"),
        ))

    for cert in parsed.get("certifications", []):
        db.add(Certification(
            candidate_id=candidate.id,
            name=cert.get("name", ""),
            issuer=cert.get("issuer"),
            date=cert.get("date"),
        ))

    for lang in parsed.get("languages", []):
        db.add(Language(
            candidate_id=candidate.id,
            name=lang.get("name", ""),
            proficiency=lang.get("proficiency"),
        ))

    await db.commit()
    await db.refresh(candidate)
    return candidate


async def _save_embedding(db: AsyncSession, candidate_id: int, parsed: dict, model_name: str):
    text = build_resume_text_for_embedding(parsed)
    vector = generate_embedding(text, model_name)
    if vector:
        emb = ResumeEmbedding(candidate_id=candidate_id, vector=vector, model_name=model_name)
        db.add(emb)
        await db.commit()


@router.post("/parse_resume", response_model=dict)
async def parse_resume_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        data = await file.read()
        raw_text = extract_resume_text(data, file.filename or "")

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

        parsed = parse_resume(raw_text)
        parsed["full_name"] = _refine_name(parsed.get("full_name"), file.filename or "")
        ats = calculate_ats_score(parsed)
        candidate = await _save_candidate(db, parsed, raw_text, ats)

        from backend.core.config import get_settings
        settings = get_settings()
        background_tasks.add_task(_save_embedding, db, candidate.id, parsed, settings.embedding_model)

        return {
            "candidate_id": candidate.id,
            "parsed": parsed,
            "ats": ats,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Resume parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_parse", response_model=List[dict])
async def batch_parse(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for file in files:
        try:
            data = await file.read()
            raw_text = extract_resume_text(data, file.filename or "")
            parsed = parse_resume(raw_text)
            parsed["full_name"] = _refine_name(parsed.get("full_name"), file.filename or "")
            ats = calculate_ats_score(parsed)
            candidate = await _save_candidate(db, parsed, raw_text, ats)

            from backend.core.config import get_settings
            settings = get_settings()
            background_tasks.add_task(_save_embedding, db, candidate.id, parsed, settings.embedding_model)

            results.append({"filename": file.filename, "candidate_id": candidate.id, "ats_score": ats["ats_score"]})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return results
