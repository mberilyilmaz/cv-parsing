import sqlite3
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent.parent  # NLP/
DB_PATH = _ROOT / "resume_parser.db"
SCHEMA_PATH = _ROOT / "src" / "db" / "schema.sql"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def save_candidate_result(result: dict, raw_text: str) -> int:
    init_db()
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO candidates
        (full_name, email, phone, address, linkedin_url, github_url, raw_text, cleaned_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("full_name"),
            result.get("email"),
            result.get("phone"),
            result.get("address"),
            result.get("linkedin_url"),
            result.get("github_url"),
            raw_text,
            result.get("cleaned_text"),
        ),
    )

    if cursor.lastrowid:
        candidate_id = cursor.lastrowid
    else:
        row = cursor.execute(
            "SELECT id FROM candidates WHERE email = ?", (result.get("email"),)
        ).fetchone()
        candidate_id = row["id"] if row else None

    for skill in set(result.get("skills", [])):
        cursor.execute(
            "INSERT INTO skills (candidate_id, skill_name, skill_type) VALUES (?, ?, ?)",
            (candidate_id, skill, "technical"),
        )

    for edu in result.get("education", []):
        cursor.execute(
            """
            INSERT INTO education (candidate_id, institution, degree, field_of_study, period, gpa)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                edu.get("institution"),
                edu.get("degree"),
                edu.get("field_of_study"),
                edu.get("period"),
                edu.get("gpa"),
            ),
        )

    for exp in result.get("experiences", []):
        cursor.execute(
            """
            INSERT INTO experiences (candidate_id, company, title, period, description, entry_type)
            VALUES (?, ?, ?, ?, ?, 'work')
            """,
            (candidate_id, exp.get("company"), exp.get("title"), exp.get("period"), exp.get("description")),
        )

    for exp in result.get("internships", []):
        cursor.execute(
            """
            INSERT INTO experiences (candidate_id, company, title, period, description, entry_type)
            VALUES (?, ?, ?, ?, ?, 'internship')
            """,
            (candidate_id, exp.get("company"), exp.get("title"), exp.get("period"), exp.get("description")),
        )

    conn.commit()
    conn.close()
    return int(candidate_id)
