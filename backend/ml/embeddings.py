"""
Resume vector embeddings using Sentence Transformers.
Supports: similarity search, candidate ranking, job matching.
"""
import json
from functools import lru_cache
from typing import List, Optional
import numpy as np
from loguru import logger


@lru_cache(maxsize=1)
def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def build_resume_text_for_embedding(parsed: dict) -> str:
    """Construct a rich text representation for embedding."""
    parts = []
    if parsed.get("full_name"):
        parts.append(parsed["full_name"])
    if parsed.get("summary"):
        parts.append(parsed["summary"])

    skills = parsed.get("skills", [])
    if skills:
        skill_names = [s["normalized_name"] if isinstance(s, dict) else s for s in skills]
        parts.append("Skills: " + ", ".join(skill_names))

    for edu in parsed.get("education", []):
        bits = [edu.get("degree"), edu.get("field_of_study"), edu.get("institution")]
        parts.append(" ".join(b for b in bits if b))

    for exp in parsed.get("experiences", []):
        bits = [exp.get("job_title"), exp.get("company"), exp.get("description", "")[:200]]
        parts.append(" ".join(b for b in bits if b))

    for proj in parsed.get("projects", []):
        if proj.get("description"):
            parts.append(proj["description"][:200])

    for cert in parsed.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " | ".join(p for p in parts if p)


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
    model = _load_model(model_name)
    if model is None:
        return None
    try:
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_candidates_by_similarity(
    job_embedding: List[float],
    candidate_embeddings: List[dict],  # [{"candidate_id": int, "vector": [...]}]
) -> List[dict]:
    scored = []
    for item in candidate_embeddings:
        if item.get("vector"):
            score = cosine_similarity(job_embedding, item["vector"])
            scored.append({**item, "similarity_score": round(score * 100, 2)})
    return sorted(scored, key=lambda x: x["similarity_score"], reverse=True)
