"""
Resume Category Classifier — training pipeline.

Trains a TF-IDF + Logistic Regression model that predicts a candidate's
job category (e.g. ENGINEERING, HR, FINANCE) from raw resume text.

Dataset: train.csv  (columns: category, skills, education, experience, text)
Outputs:
    backend/ml/models/category_clf.joblib   — trained pipeline
    backend/ml/models/metrics.json           — precision / recall / f1 / accuracy

Run:
    python -m backend.ml.train_classifier
"""
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix,
)
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "train.csv"
MODEL_DIR = ROOT / "backend" / "ml" / "models"
MODEL_PATH = MODEL_DIR / "category_clf.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def _build_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine the most informative text columns into one document per resume."""
    parts = []
    for col in ("skills", "experience", "education", "text"):
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    corpus = parts[0]
    for p in parts[1:]:
        corpus = corpus.str.cat(p, sep=" ")
    return corpus


def load_data() -> Tuple[pd.Series, pd.Series]:
    logger.info(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["category"])
    X = _build_corpus(df)
    y = df["category"].astype(str)
    logger.info(f"Loaded {len(df)} resumes across {y.nunique()} categories")
    return X, y


def build_model() -> Pipeline:
    """TF-IDF features + multinomial Logistic Regression classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
        )),
    ])


def train_and_evaluate() -> dict:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_model()
    logger.info("Training model...")
    model.fit(X_train, y_train)

    logger.info("Evaluating on held-out test set...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "precision_macro":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":          round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_classes":         int(y.nunique()),
        "classes":           sorted(y.unique().tolist()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Metrics saved to {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("RESUME CATEGORY CLASSIFIER — EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"Precision (macro) : {metrics['precision_macro']*100:.2f}%")
    print(f"Recall    (macro) : {metrics['recall_macro']*100:.2f}%")
    print(f"F1 Score  (macro) : {metrics['f1_macro']*100:.2f}%")
    print(f"F1 Score  (weighted): {metrics['f1_weighted']*100:.2f}%")
    print("=" * 60)
    print("\nPer-category report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
