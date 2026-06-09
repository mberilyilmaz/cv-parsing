import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

from src.db.sqlite_client import get_conn, save_candidate_result
from src.pdf_extractor import extract_resume_text
from src.pipeline import parse_resume_pipeline
from src.visualization import plot_skill_frequency

st.set_page_config(
    page_title="CV Ayrıştırma Sistemi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #FAFBFD;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #111827;
    }

    /* ── Üst navbar ── */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 28px 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 36px;
    }
    .topbar-left { display: flex; align-items: center; gap: 14px; }
    .topbar-icon {
        width: 40px; height: 40px;
        background: #111827;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
    }
    .topbar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.3px;
    }
    .topbar-sub {
        font-size: 0.78rem;
        color: #6B7280;
        margin-top: 1px;
    }
    .topbar-badge {
        background: #F3F4F6;
        color: #374151;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 6px;
        border: 1px solid #E5E7EB;
    }

    /* ── Upload alanı ── */
    .upload-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #D1D5DB;
        border-radius: 12px;
        padding: 10px 16px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: #111827; }

    /* ── İstatistik şeridi ── */
    .stat-strip {
        display: flex;
        gap: 12px;
        margin: 28px 0 32px 0;
    }
    .stat-item {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .stat-icon {
        font-size: 1.3rem;
        width: 38px; height: 38px;
        background: #F9FAFB;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .stat-num  { font-size: 1.5rem; font-weight: 700; color: #111827; line-height: 1; }
    .stat-lbl  { font-size: 0.73rem; color: #9CA3AF; font-weight: 500; margin-top: 2px; }

    /* ── Kart ── */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F3F4F6;
    }

    /* ── Bilgi satırı ── */
    .info-row {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 8px 0; border-bottom: 1px solid #F9FAFB; font-size: 0.88rem;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #9CA3AF; width: 100px; flex-shrink: 0; font-weight: 500; font-size: 0.84rem; }
    .info-value { color: #111827; font-weight: 600; word-break: break-all; }
    .info-value a { color: #111827; text-decoration: underline; text-underline-offset: 3px; }

    /* ── Eğitim bloğu ── */
    .edu-block { padding: 12px 0; border-bottom: 1px solid #F3F4F6; }
    .edu-block:last-child { border-bottom: none; }
    .edu-school { font-weight: 600; color: #111827; font-size: 0.92rem; }
    .edu-row    { display: flex; align-items: center; gap: 8px; margin-top: 4px; flex-wrap: wrap; }
    .edu-degree {
        background: #F3F4F6; color: #374151;
        border-radius: 5px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 600;
    }
    .edu-field  { color: #6B7280; font-size: 0.83rem; }
    .edu-period { color: #9CA3AF; font-size: 0.8rem; margin-left: auto; }
    .edu-gpa    { color: #6B7280; font-size: 0.8rem; }

    /* ── Deneyim bloğu ── */
    .exp-block  { padding: 12px 0; border-bottom: 1px solid #F3F4F6; }
    .exp-block:last-child { border-bottom: none; }
    .exp-top    { display: flex; justify-content: space-between; align-items: flex-start; }
    .exp-company { font-weight: 600; color: #111827; font-size: 0.92rem; }
    .exp-period  { color: #9CA3AF; font-size: 0.79rem; white-space: nowrap; margin-left: 8px; }
    .exp-title   { color: #6B7280; font-size: 0.83rem; margin-top: 3px; }
    .exp-desc    { color: #9CA3AF; font-size: 0.81rem; margin-top: 5px; line-height: 1.4; }

    /* ── Etiketler ── */
    .tag {
        display: inline-block;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px 3px 3px 0;
    }
    .tag-skill { background: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; }
    .tag-lang  { background: #F0FDF4; color: #15803D; border: 1px solid #DCFCE7; }
    .tag-cert  { background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA; }

    /* ── Proje maddesi ── */
    .proj-item {
        padding: 8px 0;
        border-bottom: 1px solid #F3F4F6;
        font-size: 0.86rem;
        color: #374151;
        line-height: 1.5;
    }
    .proj-item:last-child { border-bottom: none; }

    /* ── Kaydet butonu ── */
    .stButton > button {
        background: #111827 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        width: 100%;
        transition: background 0.15s !important;
    }
    .stButton > button:hover {
        background: #1F2937 !important;
    }

    /* ── Bölüm başlığı ── */
    .section-heading {
        font-size: 0.7rem;
        font-weight: 700;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 36px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #E5E7EB;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        margin-bottom: 10px;
    }
    summary { font-size: 0.85rem !important; font-weight: 600 !important; color: #374151 !important; }

    hr { border-color: #E5E7EB; margin: 0; }
    [data-testid="collapsedControl"] { display: none; }
    .stSpinner { color: #111827; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Navbar ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon">📋</div>
            <div>
                <div class="topbar-title">CV Ayrıştırma Sistemi</div>
                <div class="topbar-sub">PDF özgeçmişlerini otomatik olarak analiz edin</div>
            </div>
        </div>
        <div class="topbar-badge">v1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Yükleme alanı ───────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">CV Dosyası</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "PDF yükle",
    type=["pdf"],
    label_visibility="collapsed",
)

# ── Sonuçlar ────────────────────────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("İşleniyor..."):
        pdf_bytes = uploaded.read()
        raw_text  = extract_resume_text(pdf_bytes)

    if not raw_text.strip():
        st.error("PDF'den metin çıkarılamadı. Taranmış PDF için OCR gerekir.")
    else:
        result = parse_resume_pipeline(raw_text)

        name        = result.get("full_name") or "-"
        email       = result.get("email") or "-"
        phone       = result.get("phone") or "-"
        linkedin    = result.get("linkedin_url") or "-"
        github      = result.get("github_url") or "-"
        skills      = result.get("skills", [])
        education   = result.get("education", [])
        experiences = result.get("experiences", [])
        internships = result.get("internships", [])
        languages   = result.get("languages", [])
        projects    = result.get("projects", [])
        certs       = result.get("certifications", [])

        # ── İstatistik şeridi ───────────────────────────────────────────────
        stats = [
            ("📚", len(skills),      "Beceri"),
            ("🎓", len(education),   "Eğitim"),
            ("💼", len(experiences), "Deneyim"),
            ("📝", len(internships), "Staj"),
            ("🗂", len(projects),    "Proje"),
        ]
        cols = st.columns(5)
        for col, (icon, num, lbl) in zip(cols, stats):
            col.markdown(
                f"""<div class="stat-item">
                        <div class="stat-icon">{icon}</div>
                        <div><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>
                    </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── İki sütun ───────────────────────────────────────────────────────
        col_l, col_r = st.columns([3, 2], gap="large")

        with col_l:
            # Kişisel bilgiler
            li_val = f'<a href="{linkedin}" target="_blank">{linkedin}</a>' if linkedin != "-" else "-"
            gh_val = f'<a href="{github}"   target="_blank">{github}</a>'   if github  != "-" else "-"
            st.markdown(
                f"""<div class="card">
                    <div class="card-title">Kişisel Bilgiler</div>
                    <div class="info-row"><span class="info-label">Ad Soyad</span><span class="info-value">{name}</span></div>
                    <div class="info-row"><span class="info-label">E-posta</span><span class="info-value">{email}</span></div>
                    <div class="info-row"><span class="info-label">Telefon</span><span class="info-value">{phone}</span></div>
                    <div class="info-row"><span class="info-label">LinkedIn</span><span class="info-value">{li_val}</span></div>
                    <div class="info-row"><span class="info-label">GitHub</span><span class="info-value">{gh_val}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Eğitim
            if education:
                edu_html = ""
                for e in education:
                    deg    = f'<span class="edu-degree">{e["degree"]}</span>' if e.get("degree") else ""
                    field  = f'<span class="edu-field">{e["field_of_study"]}</span>' if e.get("field_of_study") else ""
                    period = f'<span class="edu-period">{e["period"]}</span>' if e.get("period") else ""
                    gpa    = f'<span class="edu-gpa">GPA {e["gpa"]}</span>' if e.get("gpa") else ""
                    edu_html += f"""<div class="edu-block">
                        <div class="edu-school">{e.get('institution','')}</div>
                        <div class="edu-row">{deg}{field}{gpa}{period}</div>
                    </div>"""
                st.markdown(
                    f'<div class="card"><div class="card-title">Eğitim</div>{edu_html}</div>',
                    unsafe_allow_html=True,
                )

            # İş Deneyimi
            if experiences:
                exp_html = ""
                for e in experiences:
                    exp_html += f"""<div class="exp-block">
                        <div class="exp-top">
                            <span class="exp-company">{e.get('company','')}</span>
                            <span class="exp-period">{e.get('period') or ''}</span>
                        </div>
                        <div class="exp-title">{e.get('title') or ''}</div>
                        {'<div class="exp-desc">' + e["description"] + '</div>' if e.get("description") else ''}
                    </div>"""
                st.markdown(
                    f'<div class="card"><div class="card-title">İş Deneyimi</div>{exp_html}</div>',
                    unsafe_allow_html=True,
                )

            # Staj
            if internships:
                int_html = ""
                for e in internships:
                    int_html += f"""<div class="exp-block">
                        <div class="exp-top">
                            <span class="exp-company">{e.get('company','')}</span>
                            <span class="exp-period">{e.get('period') or ''}</span>
                        </div>
                        <div class="exp-title">{e.get('title') or ''}</div>
                    </div>"""
                st.markdown(
                    f'<div class="card"><div class="card-title">Staj Deneyimi</div>{int_html}</div>',
                    unsafe_allow_html=True,
                )

        with col_r:
            # Beceriler
            if skills:
                tags = "".join(f'<span class="tag tag-skill">{s}</span>' for s in skills)
                st.markdown(
                    f'<div class="card"><div class="card-title">Beceriler</div>{tags}</div>',
                    unsafe_allow_html=True,
                )

            # Diller
            if languages:
                tags = "".join(f'<span class="tag tag-lang">{l}</span>' for l in languages)
                st.markdown(
                    f'<div class="card"><div class="card-title">Diller</div>{tags}</div>',
                    unsafe_allow_html=True,
                )

            # Sertifikalar
            if certs:
                tags = "".join(f'<span class="tag tag-cert">{c}</span>' for c in certs)
                st.markdown(
                    f'<div class="card"><div class="card-title">Sertifikalar</div>{tags}</div>',
                    unsafe_allow_html=True,
                )

            # Projeler
            if projects:
                proj_html = "".join(
                    f'<div class="proj-item">{p[:150]}</div>' for p in projects
                )
                st.markdown(
                    f'<div class="card"><div class="card-title">Projeler</div>{proj_html}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Veritabanına Kaydet", use_container_width=True):
                candidate_id = save_candidate_result(result, raw_text)
                st.success(f"Kaydedildi — Aday No: {candidate_id}")

        # ── Ham veriler ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Ham JSON Çıktısı"):
            st.json(result)
        with st.expander("Ham Metin (PDF'den çıkarılan)"):
            st.text_area("", raw_text, height=220, label_visibility="collapsed")

        from src.entities.section_parser import split_into_sections as _sps
        _secs = _sps(raw_text)
        with st.expander("Tespit Edilen Bölümler — Debug"):
            for k, v in _secs.items():
                st.markdown(f"**{k}** — {len(v)} karakter")
                st.text(v[:300] if v else "(boş)")

# ── Skill Analizi ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Toplu Beceri Analizi</div>', unsafe_allow_html=True)

if st.button("Grafiği Oluştur", use_container_width=False):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT skill_name, COUNT(*) as count FROM skills GROUP BY skill_name ORDER BY count DESC",
        conn,
    )
    conn.close()
    if df.empty:
        st.info("Henüz kayıtlı veri yok.")
    else:
        fig = plot_skill_frequency(df)
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FAFBFD",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#111827"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
