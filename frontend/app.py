"""
AI Resume Parsing & Candidate Matching Platform — Streamlit Frontend
Connects to FastAPI backend at localhost:8000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Resume Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: #FAFBFD; font-family: 'Inter', sans-serif; color: #111827;
}
.topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding-bottom:24px; border-bottom:1px solid #E5E7EB; margin-bottom:28px;
}
.topbar-title { font-size:1.2rem; font-weight:700; color:#111827; }
.topbar-sub   { font-size:0.78rem; color:#6B7280; }
.card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:20px 24px; margin-bottom:16px; }
.card-title { font-size:0.68rem; font-weight:700; color:#9CA3AF; text-transform:uppercase;
              letter-spacing:.8px; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #F3F4F6; }
.info-row { display:flex; align-items:flex-start; gap:12px; padding:7px 0;
            border-bottom:1px solid #F9FAFB; font-size:0.87rem; }
.info-row:last-child { border-bottom:none; }
.info-label { color:#9CA3AF; width:110px; flex-shrink:0; font-weight:500; }
.info-value { color:#111827; font-weight:600; word-break:break-all; }
.tag { display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.77rem;
       font-weight:600; margin:3px 3px 3px 0; }
.tag-skill { background:#F3F4F6; color:#374151; border:1px solid #E5E7EB; }
.tag-lang  { background:#F0FDF4; color:#15803D; border:1px solid #DCFCE7; }
.tag-cert  { background:#FFF7ED; color:#C2410C; border:1px solid #FED7AA; }
.tag-miss  { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; }
.tag-match { background:#F0FDF4; color:#15803D; border:1px solid #DCFCE7; }
.score-big { font-size:2.5rem; font-weight:800; color:#111827; line-height:1; }
.score-label { font-size:.72rem; color:#9CA3AF; font-weight:600; text-transform:uppercase; letter-spacing:.4px; }
.stat-card { background:#fff; border:1px solid #E5E7EB; border-radius:10px;
             padding:16px 20px; text-align:center; }
.strength  { background:#F0FDF4; border-left:3px solid #16A34A; padding:8px 12px;
             border-radius:0 8px 8px 0; margin:4px 0; font-size:.85rem; color:#15803D; }
.weakness  { background:#FEF2F2; border-left:3px solid #DC2626; padding:8px 12px;
             border-radius:0 8px 8px 0; margin:4px 0; font-size:.85rem; color:#B91C1C; }
.stButton>button { background:#111827!important; color:#fff!important; border:none!important;
                   border-radius:8px!important; font-weight:600!important; }
.stButton>button:hover { background:#1F2937!important; }
[data-testid="stFileUploader"] { background:#fff; border:1.5px dashed #D1D5DB; border-radius:12px; }
[data-testid="stExpander"] { background:#fff; border:1px solid #E5E7EB!important; border-radius:10px!important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AI Resume Platform")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Resume Upload & Parse",
        "Candidate Dashboard",
        "Job Matching",
        "Skill Analytics",
        "ATS Analytics",
    ], label_visibility="collapsed")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-title">AI Resume Parsing & Candidate Matching</div>
    <div class="topbar-sub">Powered by spaCy · Sentence Transformers · FastAPI</div>
  </div>
  <span style="background:#F3F4F6;color:#374151;font-size:.75rem;font-weight:600;
               padding:5px 12px;border-radius:6px;border:1px solid #E5E7EB;">v2.0</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Resume Upload & Parse
# ═══════════════════════════════════════════════════════════════════════════
if page == "Resume Upload & Parse":
    st.markdown("### Upload Resume")

    tab_single, tab_batch = st.tabs(["Single Upload", "Batch Upload"])

    with tab_single:
        uploaded = st.file_uploader("Drop PDF / Image here", type=["pdf", "png", "jpg", "jpeg"])

        if uploaded:
            with st.spinner("Parsing resume with AI pipeline..."):
                resp = requests.post(
                    f"{API_BASE}/resume/parse_resume",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )

            if resp.status_code != 200:
                st.error(f"API Error: {resp.text}")
            else:
                data = resp.json()
                parsed = data["parsed"]
                ats    = data["ats"]
                cid    = data["candidate_id"]

                st.success(f"Parsed successfully — Candidate ID: {cid}")

                # ── Trained ML model prediction
                pred = parsed.get("predicted_category")
                if pred:
                    conf = pred.get("confidence", 0) * 100
                    top3 = ", ".join(f"{t['category']} ({t['score']*100:.0f}%)" for t in pred.get("top3", []))
                    st.info(
                        f"🧠 **Predicted Job Category (trained ML model):** "
                        f"**{pred['category']}** — confidence {conf:.1f}%  \n"
                        f"Top-3: {top3}"
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Stats strip
                cols = st.columns(5)
                for col, (num, lbl) in zip(cols, [
                    (len(parsed.get("skills", [])), "Skills"),
                    (len(parsed.get("education", [])), "Education"),
                    (len(parsed.get("experiences", [])), "Experience"),
                    (len(parsed.get("certifications", [])), "Certifications"),
                    (len(parsed.get("projects", [])), "Projects"),
                ]):
                    col.markdown(f'<div class="stat-card"><div class="score-big">{num}</div>'
                                 f'<div class="score-label">{lbl}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_l, col_r = st.columns([3, 2], gap="large")

                with col_l:
                    # Personal info
                    li = parsed.get("linkedin_url") or "-"
                    gh = parsed.get("github_url") or "-"
                    li_html = f'<a href="{li}">{li}</a>' if li != "-" else "-"
                    gh_html = f'<a href="{gh}">{gh}</a>' if gh != "-" else "-"
                    st.markdown(f"""<div class="card"><div class="card-title">Personal Information</div>
                        <div class="info-row"><span class="info-label">Full Name</span><span class="info-value">{parsed.get('full_name') or '-'}</span></div>
                        <div class="info-row"><span class="info-label">Email</span><span class="info-value">{parsed.get('email') or '-'}</span></div>
                        <div class="info-row"><span class="info-label">Phone</span><span class="info-value">{parsed.get('phone') or '-'}</span></div>
                        <div class="info-row"><span class="info-label">Location</span><span class="info-value">{parsed.get('location') or '-'}</span></div>
                        <div class="info-row"><span class="info-label">LinkedIn</span><span class="info-value">{li_html}</span></div>
                        <div class="info-row"><span class="info-label">GitHub</span><span class="info-value">{gh_html}</span></div>
                        <div class="info-row"><span class="info-label">Experience</span><span class="info-value">{parsed.get('total_experience_years', 0)} years</span></div>
                    </div>""", unsafe_allow_html=True)

                    # Education
                    if parsed.get("education"):
                        edu_html = "".join(
                            f'<div style="padding:10px 0;border-bottom:1px solid #F3F4F6;">'
                            f'<div style="font-weight:600">{e.get("institution","")}</div>'
                            f'<div style="color:#6B7280;font-size:.83rem">'
                            f'{e.get("degree","") or ""} {e.get("field_of_study","") or ""} {e.get("period","") or ""}</div></div>'
                            for e in parsed["education"]
                        )
                        st.markdown(f'<div class="card"><div class="card-title">Education</div>{edu_html}</div>', unsafe_allow_html=True)

                    # Experience
                    if parsed.get("experiences"):
                        exp_html = "".join(
                            f'<div style="padding:10px 0;border-bottom:1px solid #F3F4F6;">'
                            f'<div style="display:flex;justify-content:space-between;">'
                            f'<b>{e.get("company","")}</b><span style="color:#9CA3AF;font-size:.8rem">{e.get("period","") or ""}</span></div>'
                            f'<div style="color:#6B7280;font-size:.83rem">{e.get("job_title","") or ""}</div></div>'
                            for e in parsed["experiences"]
                        )
                        st.markdown(f'<div class="card"><div class="card-title">Experience</div>{exp_html}</div>', unsafe_allow_html=True)

                with col_r:
                    # ATS Score
                    score = ats["ats_score"]
                    color = "#16A34A" if score >= 75 else "#D97706" if score >= 50 else "#DC2626"
                    st.markdown(f"""<div class="card" style="text-align:center">
                        <div class="card-title">ATS Score</div>
                        <div style="font-size:3rem;font-weight:800;color:{color}">{score}</div>
                        <div style="color:#6B7280;font-size:.85rem;margin-top:4px">{ats.get('recommendation','')}</div>
                    </div>""", unsafe_allow_html=True)

                    # Score breakdown
                    bd = ats.get("breakdown", {})
                    fig = go.Figure(go.Bar(
                        x=[v["score"] for v in bd.values()],
                        y=list(bd.keys()),
                        orientation="h",
                        marker_color="#111827",
                    ))
                    fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                                      paper_bgcolor="white", plot_bgcolor="white",
                                      xaxis=dict(range=[0,100], ticksuffix="%"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Skills
                    if parsed.get("skills"):
                        tags = "".join(f'<span class="tag tag-skill">{s["normalized_name"] if isinstance(s,dict) else s}</span>'
                                       for s in parsed["skills"])
                        st.markdown(f'<div class="card"><div class="card-title">Skills</div>{tags}</div>', unsafe_allow_html=True)

                    # Strengths / Weaknesses
                    strengths = ats.get("strengths", [])
                    weaknesses = ats.get("weaknesses", [])
                    if strengths or weaknesses:
                        s_html = "".join(f'<div class="strength">✓ {s}</div>' for s in strengths)
                        w_html = "".join(f'<div class="weakness">✗ {w}</div>' for w in weaknesses)
                        st.markdown(f'<div class="card"><div class="card-title">Explainability</div>{s_html}{w_html}</div>', unsafe_allow_html=True)

                # Raw JSON
                with st.expander("Full JSON Output"):
                    st.json(data)

    with tab_batch:
        files = st.file_uploader("Upload multiple resumes", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)
        if files and st.button("Parse All", use_container_width=True):
            with st.spinner(f"Parsing {len(files)} resumes..."):
                resp = requests.post(
                    f"{API_BASE}/resume/batch_parse",
                    files=[("files", (f.name, f.getvalue(), f.type)) for f in files],
                )
            if resp.status_code == 200:
                results = resp.json()
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
            else:
                st.error(resp.text)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Candidate Dashboard
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Candidate Dashboard":
    st.markdown("### Candidate Dashboard")

    # Show any pending flash message (set before a rerun)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    def load_all_candidates(filters: dict | None = None):
        params = {"limit": 200}
        if filters:
            params.update(filters)
        try:
            r = requests.get(f"{API_BASE}/candidates/candidate_search", params=params, timeout=15)
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            st.error(f"Could not reach API: {e}")
            return []

    # ── Filters ─────────────────────────────────────────────────────────────
    with st.expander("Search & Filter", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        skill_filter = c1.text_input("Skill contains")
        min_exp      = c2.number_input("Min experience (yrs)", 0.0, 50.0, 0.0)
        max_exp      = c3.number_input("Max experience (yrs)", 0.0, 50.0, 50.0)
        min_ats      = c4.slider("Min ATS score", 0, 100, 0)

    active_filters = {"min_experience": min_exp, "max_experience": max_exp, "min_ats": min_ats}
    if skill_filter:
        active_filters["skill"] = skill_filter

    candidates = load_all_candidates(active_filters)

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Candidates", len(candidates))
    if candidates:
        avg_ats = sum(c.get("ats_score", 0) for c in candidates) / len(candidates)
        avg_exp = sum(c.get("total_experience_years", 0) for c in candidates) / len(candidates)
        m2.metric("Avg ATS Score", f"{avg_ats:.1f}")
        m3.metric("Avg Experience", f"{avg_exp:.1f} yrs")

    st.markdown("<br>", unsafe_allow_html=True)

    if not candidates:
        st.info("No candidates in the database yet. Upload resumes from the 'Resume Upload & Parse' page.")
    else:
        # ── Full candidate table ─────────────────────────────────────────────
        df = pd.DataFrame(candidates)[["id", "full_name", "email", "location",
                                       "total_experience_years", "ats_score", "created_at"]]
        df.columns = ["ID", "Name", "Email", "Location", "Experience (yrs)", "ATS Score", "Created"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Export CSV",
            data=requests.get(f"{API_BASE}/candidates/export/csv").content,
            file_name="candidates.csv", mime="text/csv",
        )

        st.markdown("---")

        # ── Manage a single candidate ────────────────────────────────────────
        st.markdown("#### Manage Candidate")
        options = {f"#{c['id']} — {c.get('full_name') or 'Unknown'} ({c.get('email') or 'no email'})": c["id"]
                   for c in candidates}
        selected_label = st.selectbox("Select a candidate", list(options.keys()))
        selected_id = options[selected_label]

        col_view, col_del = st.columns([1, 1])

        with col_view:
            if st.button("View Details", use_container_width=True):
                r = requests.get(f"{API_BASE}/candidates/{selected_id}")
                if r.status_code == 200:
                    st.json(r.json())
                else:
                    st.error("Candidate not found.")

        with col_del:
            if st.button("🗑 Delete This Candidate", use_container_width=True):
                st.session_state["confirm_delete"] = selected_id

        # Confirmation
        if st.session_state.get("confirm_delete") == selected_id:
            st.warning(f"Permanently delete candidate #{selected_id} ({selected_label})?")
            cc1, cc2 = st.columns([1, 1])
            if cc1.button("✓ Yes, delete", use_container_width=True):
                r = requests.delete(f"{API_BASE}/candidates/{selected_id}")
                st.session_state["confirm_delete"] = None
                if r.status_code == 200:
                    st.session_state["flash"] = f"Candidate #{selected_id} deleted successfully."
                else:
                    st.session_state["flash"] = f"Delete failed: {r.text}"
                st.rerun()  # reload list → count drops, row disappears
            if cc2.button("✗ Cancel", use_container_width=True):
                st.session_state["confirm_delete"] = None
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Job Matching
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Job Matching":
    st.markdown("### Job Matching Engine")

    col_jd, col_res = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("#### Job Description")
        job_title = st.text_input("Job Title", "Senior Python Developer")
        jd_text = st.text_area("Paste job description", height=280,
            placeholder="We are looking for a Python developer with experience in FastAPI, PostgreSQL, Docker...")
        required_skills = st.text_input("Required skills (comma separated)", "Python, FastAPI, PostgreSQL, Docker")
        required_years  = st.number_input("Min experience (years)", 0.0, 20.0, 2.0)
        top_n = st.slider("Top N candidates", 3, 20, 5)

    with col_res:
        st.markdown("#### Results")
        if st.button("Match Candidates", use_container_width=True):
            if not jd_text.strip():
                st.warning("Please paste a job description.")
            else:
                payload = {
                    "job_title": job_title,
                    "description": jd_text,
                    "required_skills": [s.strip() for s in required_skills.split(",") if s.strip()],
                    "required_years": required_years,
                }
                with st.spinner("Ranking candidates..."):
                    resp = requests.post(f"{API_BASE}/jobs/top_candidates?top_n={top_n}", json=payload)

                if resp.status_code == 200:
                    results = resp.json()
                    if not results:
                        st.info("No candidates in database yet. Upload some resumes first.")
                    else:
                        for i, r in enumerate(results, 1):
                            score = r.get("final_score", 0)
                            color = "#16A34A" if score >= 75 else "#D97706" if score >= 50 else "#DC2626"
                            matched = ", ".join(r.get("matched_skills", [])[:5])
                            missing = ", ".join(r.get("missing_skills", [])[:5])
                            st.markdown(f"""<div class="card">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <div><b>#{i} {r.get('full_name','N/A')}</b><div style="color:#6B7280;font-size:.82rem">{r.get('email','')}</div></div>
                                    <div style="font-size:1.8rem;font-weight:800;color:{color}">{score}</div>
                                </div>
                                <div style="margin-top:10px;font-size:.82rem">
                                    {"<span class='tag tag-match'>✓ " + "</span><span class='tag tag-match'>✓ ".join(r.get('matched_skills',[])[:5]) + "</span>" if r.get('matched_skills') else ""}
                                    {"<span class='tag tag-miss'>✗ " + "</span><span class='tag tag-miss'>✗ ".join(r.get('missing_skills',[])[:3]) + "</span>" if r.get('missing_skills') else ""}
                                </div>
                                <div style="color:#6B7280;font-size:.8rem;margin-top:6px">{r.get('recommendation','')}</div>
                            </div>""", unsafe_allow_html=True)
                else:
                    st.error(resp.text)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Skill Analytics
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Skill Analytics":
    st.markdown("### Skill Analytics")

    resp = requests.get(f"{API_BASE}/candidates/candidate_search", params={"limit": 200})
    if resp.status_code != 200:
        st.error("Could not load candidates.")
    else:
        candidates = resp.json()
        if not candidates:
            st.info("No candidates in database yet.")
        else:
            all_skills = []
            for c in candidates:
                cid = c["id"]
                detail = requests.get(f"{API_BASE}/candidates/{cid}")
                if detail.status_code == 200:
                    for s in detail.json().get("skills", []):
                        all_skills.append({"skill": s["name"], "category": s.get("category","other")})

            if all_skills:
                df = pd.DataFrame(all_skills)
                skill_counts = df["skill"].value_counts().reset_index()
                skill_counts.columns = ["skill", "count"]

                fig = px.bar(skill_counts.head(25), x="count", y="skill", orientation="h",
                             title="Top 25 Skills Across All Candidates",
                             color="count", color_continuous_scale="Blues")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=500)
                st.plotly_chart(fig, use_container_width=True)

                cat_counts = df["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig2 = px.pie(cat_counts, values="count", names="category", title="Skill Category Distribution")
                fig2.update_layout(paper_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — ATS Analytics
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ATS Analytics":
    st.markdown("### ATS Analytics")

    resp = requests.get(f"{API_BASE}/candidates/candidate_search", params={"limit": 200})
    if resp.status_code != 200:
        st.error("Could not load candidates.")
    else:
        candidates = resp.json()
        if not candidates:
            st.info("No candidates in database yet.")
        else:
            df = pd.DataFrame(candidates)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Candidates", len(df))
            c2.metric("Avg ATS Score", f"{df['ats_score'].mean():.1f}")
            c3.metric("Avg Experience", f"{df['total_experience_years'].mean():.1f} yrs")

            fig = px.histogram(df, x="ats_score", nbins=20,
                               title="ATS Score Distribution", color_discrete_sequence=["#111827"])
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFBFD")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(df, x="total_experience_years", y="ats_score",
                              hover_data=["full_name","email"],
                              title="Experience vs ATS Score", color="ats_score",
                              color_continuous_scale="Greys")
            fig2.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFBFD")
            st.plotly_chart(fig2, use_container_width=True)
