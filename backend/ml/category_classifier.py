"""
Resume Category Classifier — inference.

Loads the trained TF-IDF + Logistic Regression model and predicts the
job category for a resume, with a confidence score.

The model is produced by backend/ml/train_classifier.py.
If the model file is missing, prediction returns None (system degrades gracefully).
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

MODEL_PATH = Path(__file__).resolve().parent / "models" / "category_clf.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        logger.warning(f"Category model not found at {MODEL_PATH}. "
                       f"Run: python -m backend.ml.train_classifier")
        return None
    try:
        import joblib
        logger.info("Loading resume category classifier...")
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load category model: {e}")
        return None


def predict_category(resume_text: str) -> Optional[Dict[str, Any]]:
    """
    Predict the job category of a resume.
    Returns {"category": str, "confidence": float, "top3": [...]} or None.
    """
    model = _load_model()
    if model is None or not resume_text.strip():
        return None
    try:
        probs = model.predict_proba([resume_text])[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        return {
            "category":   ranked[0][0],
            "confidence": round(float(ranked[0][1]), 4),
            "top3": [{"category": c, "score": round(float(p), 4)} for c, p in ranked[:3]],
        }
    except Exception as e:
        logger.error(f"Category prediction failed: {e}")
        return None
