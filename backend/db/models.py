from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Integer, String, Float, Text, DateTime, ForeignKey,
    JSON, Boolean, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.database import Base
import enum


class DegreeLevel(str, enum.Enum):
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    OTHER = "other"


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    email: Mapped[Optional[str]] = mapped_column(String(200), unique=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50))
    location: Mapped[Optional[str]] = mapped_column(String(200))
    linkedin_url: Mapped[Optional[str]] = mapped_column(String(500))
    github_url: Mapped[Optional[str]] = mapped_column(String(500))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[Optional[str]] = mapped_column(Text)
    total_experience_years: Mapped[float] = mapped_column(Float, default=0.0)
    ats_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    skills: Mapped[List["Skill"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    experiences: Mapped[List["Experience"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    education: Mapped[List["Education"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    projects: Mapped[List["Project"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    certifications: Mapped[List["Certification"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    languages: Mapped[List["Language"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    embedding: Mapped[Optional["ResumeEmbedding"]] = relationship(back_populates="candidate", cascade="all, delete-orphan", uselist=False)


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    raw_name: Mapped[str] = mapped_column(String(200))
    normalized_name: Mapped[str] = mapped_column(String(200), index=True)
    category: Mapped[Optional[str]] = mapped_column(String(100))  # programming, framework, tool, soft
    is_implicit: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    candidate: Mapped["Candidate"] = relationship(back_populates="skills")


class Experience(Base):
    __tablename__ = "experiences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    company: Mapped[Optional[str]] = mapped_column(String(300))
    job_title: Mapped[Optional[str]] = mapped_column(String(300))
    location: Mapped[Optional[str]] = mapped_column(String(200))
    period: Mapped[Optional[str]] = mapped_column(String(100))
    start_date: Mapped[Optional[str]] = mapped_column(String(50))
    end_date: Mapped[Optional[str]] = mapped_column(String(50))
    duration_months: Mapped[Optional[int]] = mapped_column(Integer)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    entry_type: Mapped[str] = mapped_column(String(20), default="work")  # work | internship

    candidate: Mapped["Candidate"] = relationship(back_populates="experiences")


class Education(Base):
    __tablename__ = "education"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    institution: Mapped[Optional[str]] = mapped_column(String(300))
    degree: Mapped[Optional[str]] = mapped_column(String(200))
    degree_level: Mapped[Optional[str]] = mapped_column(String(50))
    field_of_study: Mapped[Optional[str]] = mapped_column(String(200))
    period: Mapped[Optional[str]] = mapped_column(String(100))
    start_date: Mapped[Optional[str]] = mapped_column(String(50))
    end_date: Mapped[Optional[str]] = mapped_column(String(50))
    gpa: Mapped[Optional[str]] = mapped_column(String(20))

    candidate: Mapped["Candidate"] = relationship(back_populates="education")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    name: Mapped[Optional[str]] = mapped_column(String(300))
    description: Mapped[Optional[str]] = mapped_column(Text)
    technologies: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    url: Mapped[Optional[str]] = mapped_column(String(500))

    candidate: Mapped["Candidate"] = relationship(back_populates="projects")


class Certification(Base):
    __tablename__ = "certifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    issuer: Mapped[Optional[str]] = mapped_column(String(200))
    date: Mapped[Optional[str]] = mapped_column(String(50))

    candidate: Mapped["Candidate"] = relationship(back_populates="certifications")


class Language(Base):
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    proficiency: Mapped[Optional[str]] = mapped_column(String(50))  # native, fluent, intermediate, basic

    candidate: Mapped["Candidate"] = relationship(back_populates="languages")


class ResumeEmbedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), unique=True, index=True)
    vector: Mapped[Optional[list]] = mapped_column(JSON)  # stored as list; use pgvector in prod
    model_name: Mapped[str] = mapped_column(String(200), default="all-MiniLM-L6-v2")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    candidate: Mapped["Candidate"] = relationship(back_populates="embedding")
