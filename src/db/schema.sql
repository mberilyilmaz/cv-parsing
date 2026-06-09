PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    email TEXT UNIQUE,
    phone TEXT,
    address TEXT,
    linkedin_url TEXT,
    github_url TEXT,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    skill_name TEXT NOT NULL,
    skill_type TEXT CHECK(skill_type IN ('general', 'technical')),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS education (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    institution TEXT,
    degree TEXT,
    field_of_study TEXT,
    period TEXT,
    gpa TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    company TEXT,
    title TEXT,
    period TEXT,
    description TEXT,
    entry_type TEXT CHECK(entry_type IN ('work', 'internship')) DEFAULT 'work',
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
);
