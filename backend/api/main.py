from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os

from backend.core.config import get_settings
from backend.core.logging_config import setup_logging
from backend.db.database import init_db
from backend.api.routers import resume, candidates, jobs

os.makedirs("logs", exist_ok=True)
setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database...")
    await init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Resume Parsing & Candidate Matching Platform",
    description="Production-ready ATS powered by spaCy, Sentence Transformers, and FastAPI.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume.router)
app.include_router(candidates.router)
app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
