# AI-Powered Resume Parsing & Candidate Matching Platform

Production-style Applicant Tracking System (ATS) that goes well beyond a
regex-based parser: multi-layer OCR, NLP entity extraction, skill intelligence,
embedding-based semantic matching, and a **trained machine-learning model**
for resume category classification.

## Architecture

```
PDF / Image  →  OCR (pdfplumber / PyMuPDF / EasyOCR)  →  NLP Pipeline  →  PostgreSQL/SQLite
                                                              ↓
Job Description  →  Requirement extraction  →  ATS + Semantic matching  →  Ranked candidates
```

- **Backend:** FastAPI (`backend/`)
- **Frontend:** Streamlit (`frontend/app.py`)
- **Database:** SQLite (default) / PostgreSQL

## Models

| Model | Purpose | Trained? |
|-------|---------|----------|
| **TF-IDF + Logistic Regression** | Resume → job category classification | ✅ **Trained by us** on `train.csv` |
| spaCy `en_core_web_lg` | Named entity recognition (name, location) | Pre-trained |
| Sentence-Transformers `all-MiniLM-L6-v2` | Resume/job embeddings (semantic match) | Pre-trained |
| RapidFuzz | Skill normalization (fuzzy matching) | Algorithm |

### Trained Model — Resume Category Classifier

We trained a supervised classifier that predicts a candidate's job category
(23 classes: ENGINEERING, HR, FINANCE, HEALTHCARE, …) from resume text.

- **Dataset:** `train.csv` (2,257 annotated resumes, 23 categories)
- **Features:** TF-IDF (uni- + bi-grams, 20k features)
- **Classifier:** Multinomial Logistic Regression (class-balanced)
- **Evaluation (held-out 20% test set):**

| Metric | Score |
|--------|-------|
| Accuracy | 68.6% |
| Precision (macro) | 69.2% |
| Recall (macro) | 64.0% |
| F1 (macro) | 64.0% |
| F1 (weighted) | 67.9% |

Train / re-train:
```bash
python -m backend.ml.train_classifier
```
Outputs `backend/ml/models/category_clf.joblib` and `metrics.json`.
The prediction is shown for every uploaded resume in the UI.

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m backend.ml.train_classifier      # train the category model
```

## Run

```bash
uvicorn backend.api.main:app --reload --port 8000   # API   (terminal 1)
streamlit run frontend/app.py                        # UI    (terminal 2)
```

## Tests

```bash
python -m pytest tests/ -v
```

## Features

- Resume input: PDF, scanned PDF, images (PNG/JPG), batch upload
- NLP: cleaning, section segmentation, entity extraction, skill extraction,
  skill normalization, candidate ranking
- Entities: PERSON, EMAIL, PHONE, LOCATION, UNIVERSITY, DEGREE, COMPANY,
  JOB_TITLE, SKILL, PROJECT, CERTIFICATION, LANGUAGE
- Skill intelligence: explicit + implicit skill inference
- Embedding-based semantic similarity search
- Job matching engine: match score, matched/missing skills, ranking
- ATS scoring with explainability (strengths / weaknesses / recommendation)
- Trained ML category classifier with precision/recall/F1 evaluation
