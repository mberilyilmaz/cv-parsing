from pydantic import BaseModel, EmailStr, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


class SkillSchema(BaseModel):
    raw_name: str
    normalized_name: str
    category: Optional[str] = None
    is_implicit: bool = False
    confidence: float = 1.0


class ExperienceSchema(BaseModel):
    company: Optional[str] = None
    job_title: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None
    is_current: bool = False
    entry_type: str = "work"


class EducationSchema(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    degree_level: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None


class ProjectSchema(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = []
    url: Optional[str] = None


class CertificationSchema(BaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None


class LanguageSchema(BaseModel):
    name: str
    proficiency: Optional[str] = None


class ParsedResumeSchema(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    summary: Optional[str] = None
    education: List[EducationSchema] = []
    experiences: List[ExperienceSchema] = []
    skills: List[SkillSchema] = []
    languages: List[LanguageSchema] = []
    projects: List[ProjectSchema] = []
    certifications: List[CertificationSchema] = []
    total_experience_years: float = 0.0
    sections_detected: List[str] = []


class CandidateResponse(BaseModel):
    id: int
    full_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    location: Optional[str]
    linkedin_url: Optional[str]
    github_url: Optional[str]
    total_experience_years: float
    ats_score: float
    skills: List[SkillSchema] = []
    education: List[EducationSchema] = []
    experiences: List[ExperienceSchema] = []
    created_at: datetime

    class Config:
        from_attributes = True


class JobRequirementSchema(BaseModel):
    job_title: str
    description: str
    required_skills: List[str] = []
    required_years: float = 0
    required_degree: Optional[str] = None
    required_certifications: List[str] = []
    required_languages: List[str] = ["English"]


class MatchResultSchema(BaseModel):
    candidate_id: int
    full_name: Optional[str]
    email: Optional[str]
    final_score: float
    ats_score: float
    semantic_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
