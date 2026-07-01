import os
import re
import csv
import json
import time
import pickle
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Embeddings
from sentence_transformers import SentenceTransformer

# Custom Scorers
from src.scorers.structural_scorer import StructuralScorer
from src.reasoning import ReasoningGenerator

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ARIA — Adaptive Ranked Intelligence Architecture",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject modern dark-theme UI with glow and glassmorphism styling
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Glowing Title & Subtitle */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #1f4068, #162447);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #FF4B4B);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.2rem;
        color: #B2BEC3;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Glassmorphism Card Wrapper */
    .glass-card {
        background: rgba(38, 39, 48, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Headers & Text */
    h2, h3 {
        color: #FF8F8F !important;
        font-weight: 600 !important;
    }
    .preview-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #FF8F8F;
    }
    
    /* Keyframe Animations */
    @keyframes shine {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caching Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Loads and caches the SentenceTransformer model once on Streamlit Cloud."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------------------------
# Scoring Helpers (Standard Parity)
# ---------------------------------------------------------------------------
def compute_days_since_active(date_str: str) -> float:
    try:
        ref_date = date(2026, 6, 27)
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        delta = (ref_date - d).days
        return max(0.0, float(delta))
    except (ValueError, TypeError):
        return 365.0

def compute_inline_behavioral_score(cand: dict) -> float:
    redrob = cand.get("redrob_signals", {})
    
    # 1. Availability cluster (notice, open_to_work, days_since_active)
    notice = redrob.get("notice_period_days", 0)
    notice_norm = max(0.0, min(1.0, 1.0 - notice / 180.0))
    open_to_work = 1.0 if redrob.get("open_to_work_flag", False) else 0.0
    days_since = compute_days_since_active(redrob.get("last_active_date", ""))
    active_norm = max(0.0, min(1.0, 1.0 - days_since / 365.0))
    
    availability = 0.30 * notice_norm + 0.35 * open_to_work + 0.35 * active_norm

    # 2. Responsiveness cluster (rate, time, interview_rate)
    resp_rate = redrob.get("recruiter_response_rate", 0.0)
    resp_time = redrob.get("avg_response_time_hours", 72.0)
    time_norm = max(0.0, min(1.0, 1.0 - resp_time / 72.0))
    interview = redrob.get("interview_completion_rate", 0.0)
    
    responsiveness = (resp_rate + time_norm + interview) / 3.0

    # 3. Platform Credibility cluster (profile_completeness, connections, endorsements, github, search, saved)
    completeness = redrob.get("profile_completeness_score", 0.0) / 100.0
    connections = min(1.0, redrob.get("connection_count", 0) / 1000.0)
    endorsements = min(1.0, redrob.get("endorsements_received", 0) / 500.0)
    github = redrob.get("github_activity_score", 0.0)
    github_norm = 0.0 if github == -1 else min(1.0, github / 100.0)
    search = min(1.0, redrob.get("search_appearance_30d", 0) / 2000.0)
    saved = min(1.0, redrob.get("saved_by_recruiters_30d", 0) / 50.0)

    credibility = (completeness + connections + endorsements + github_norm + search + saved) / 6.0

    # 4. Verification cluster (email, phone, linkedin)
    email = 1.0 if redrob.get("verified_email", False) else 0.0
    phone = 1.0 if redrob.get("verified_phone", False) else 0.0
    linkedin = 1.0 if redrob.get("linkedin_connected", False) else 0.0
    
    verification = (email + phone + linkedin) / 3.0

    # 5. Engagement cluster (views, applications, offer_acceptance)
    views = min(1.0, redrob.get("profile_views_received_30d", 0) / 1000.0)
    apps = min(1.0, redrob.get("applications_submitted_30d", 0) / 100.0)
    acceptance = redrob.get("offer_acceptance_rate", -1)
    acceptance_norm = 0.5 if acceptance == -1 else acceptance

    engagement = (views + apps + acceptance_norm) / 3.0

    # Combine clusters
    weighted_beh = (
        0.30 * responsiveness +
        0.25 * credibility +
        0.20 * availability +
        0.15 * verification +
        0.10 * engagement
    )
    return max(0.0, min(1.0, weighted_beh))

def compute_inline_integrity_score(cand: dict) -> tuple[float, list[str]]:
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    redrob = cand.get("redrob_signals", {})
    
    flags = []
    penalties = 0.0

    # 1. Skill mismatch
    mismatch_count = 0
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) < 6:
            mismatch_count += 1
    if mismatch_count > 0:
        flags.append("skill_experience_mismatch")
        penalties += min(0.45, 0.15 * mismatch_count)

    # 2. Stuffing
    career_months = sum(role.get("duration_months", 0) for role in career)
    skill_count = len(skills)
    ratio = skill_count / max(1, career_months)
    if ratio > 0.5 or skill_count > 30:
        flags.append("keyword_stuffing")
        penalties += 0.25

    # 3. Timeline check (Overlapping roles)
    dated_roles = []
    for role in career:
        try:
            start = datetime.strptime(role.get("start_date", ""), "%Y-%m-%d") if role.get("start_date") else None
            end = datetime.strptime(role.get("end_date", ""), "%Y-%m-%d") if role.get("end_date") else datetime(2026, 6, 27)
            if start:
                dated_roles.append((start, end))
        except ValueError:
            pass

    overlap_count = 0
    for i in range(len(dated_roles)):
        for j in range(i + 1, len(dated_roles)):
            s1, e1 = dated_roles[i]
            s2, e2 = dated_roles[j]
            overlap_days = (min(e1, e2) - max(s1, s2)).days
            if overlap_days > 180: # overlapping > 6 months
                overlap_count += 1
    if overlap_count > 0:
        flags.append("impossible_timeline")
        penalties += min(0.60, 0.30 * overlap_count)

    # 4. Behavioral twin
    beh_score = compute_inline_behavioral_score(cand)
    if beh_score > 0.7 and career_months < 12:
        flags.append("behavioral_twin")
        penalties += 0.20

    score = max(0.0, min(1.0, 1.0 - penalties))
    return score, flags

def compute_inline_skill_score(cand: dict, must_have_kws: list, nice_to_have_kws: list) -> float:
    skills = set(s.get("name", "").lower().strip() for s in cand.get("skills", []) if s.get("name"))
    
    # JD keywords lowercased
    jd_kws = set(k.lower().strip() for k in must_have_kws + nice_to_have_kws)
    
    # Overlap count
    matches = len(skills.intersection(jd_kws))
    
    # Normalize (max 10 matches = 1.0)
    return min(1.0, matches / 10.0)

# ---------------------------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------------------------
def main():
    # Glowing header banner
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">ARIA</h1>
        <div class="main-subtitle">Adaptive Ranked Intelligence Architecture</div>
        <div style="color:#7F8C8D; font-size:0.9rem; margin-top:0.3rem;">Candidate Ranking Sandbox & Demo | Noida / Pune Core Systems</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar documentation & instructions
    with st.sidebar:
        st.header("Pipeline Settings")
        st.info("💡 **Streamlit Cloud Sandbox Mode:** Scoring is re-computed on-the-fly dynamically. No local pre-computed files are required.")
        st.markdown("""
        **Starting Fusion Weights:**
        *   `Structural`: 0.45 (Primary resume facts)
        *   `Semantic`: 0.20 (Embedding similarity)
        *   `Skill Graph`: 0.20 (PMI skill map overlap)
        *   `Behavioral`: 0.15 (Activity & responsiveness)
        *   `Integrity`: Multiplicative penalty
        """)

    # -----------------------------------------------------------------------
    # Section 1: File Upload
    # -----------------------------------------------------------------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if "candidates" not in st.session_state:
        st.session_state["candidates"] = None

    uploaded_file = st.file_uploader(
        "Upload candidate profiles in JSON or JSONL format (Max 100 profiles for sandbox)", 
        type=["json", "jsonl"]
    )
    st.markdown('<div style="text-align: center; margin: 5px 0; color: #888;">OR</div>', unsafe_allow_html=True)
    load_sample = st.button("📂 Load Sample Candidates (sample_candidates.json)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            filename = uploaded_file.name
            file_bytes = uploaded_file.read()
            content = file_bytes.decode("utf-8").strip()
            cands = []
            if filename.endswith(".json"):
                cands = json.loads(content)
            else:
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        cands.append(json.loads(line))
            st.session_state["candidates"] = cands
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.session_state["candidates"] = None
    elif load_sample:
        sample_path = Path("data/sample_candidates.json")
        if sample_path.exists():
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.session_state["candidates"] = json.load(f)
                st.success("Successfully loaded sample_candidates.json!")
            except Exception as e:
                st.error(f"Failed to read sample file: {e}")
                st.session_state["candidates"] = None
        else:
            st.error("data/sample_candidates.json not found in repository.")
            st.session_state["candidates"] = None

    candidates = st.session_state["candidates"]
    if candidates is not None:
        total_candidates = len(candidates)
        if total_candidates == 0:
            st.warning("No candidates found.")
            return

        if total_candidates > 100:
            st.error(f"Candidate count ({total_candidates}) exceeds sandbox limit of 100. Please load a smaller batch.")
            st.session_state["candidates"] = None
            return

        # -------------------------------------------------------------------
        # Preview Card
        # -------------------------------------------------------------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="preview-header">📊 Batch Upload Preview ({total_candidates} candidates loaded)</div>', unsafe_allow_html=True)
        
        # Display preview table
        preview_data = []
        for cand in candidates[:10]:
            profile = cand.get("profile", {})
            preview_data.append({
                "ID": cand.get("candidate_id"),
                "Name": profile.get("anonymized_name", "Anonymized"),
                "Current Title": profile.get("current_title", "N/A"),
                "Experience (YOE)": profile.get("years_of_experience", 0.0),
                "Location": profile.get("location", "N/A")
            })
        st.table(pd.DataFrame(preview_data))
        if total_candidates > 100:
            st.write(f"... and {total_candidates - 10} more candidates.")
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # Section 2: Run ranking
        # -------------------------------------------------------------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Process Candidates")
        
        if st.button("🚀 Run ARIA Ranking", type="primary", use_container_width=True):
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Phase A: Loading JD structure
                status_container.markdown("⚙️ **Step 1/5:** Loading parsed Job Description structure...")
                progress_bar.progress(10)
                
                jd_path = Path("artifacts/jd_structured.json")
                if not jd_path.exists():
                    st.error("Error: artifacts/jd_structured.json not found in repository. Ensure it was committed and pushed.")
                    return
                with open(jd_path, "r", encoding="utf-8") as f:
                    jd_data = json.load(f)
                
                must_have_kws = jd_data.get("must_have_keywords", [])
                nice_to_have_kws = jd_data.get("nice_to_have_keywords", [])
                
                # JD Sections for semantic embedding
                jd_must = jd_data.get("must_have_signals", [])
                jd_ideal = jd_data.get("ideal_profile_description", "")
                jd_core = jd_data.get("core_responsibilities", "")
                
                # Phase B: Semantic embeddings (on-the-fly)
                status_container.markdown("🤖 **Step 2/5:** Embedding candidate profiles & Job Description sections...")
                progress_bar.progress(35)
                
                model = get_embedding_model()
                
                # Embed JD sections once
                emb_jd_must = model.encode(" ".join(jd_must), convert_to_numpy=True)
                emb_jd_ideal = model.encode(jd_ideal, convert_to_numpy=True)
                emb_jd_core = model.encode(jd_core, convert_to_numpy=True)
                
                # Normalize JD embeddings
                emb_jd_must /= np.linalg.norm(emb_jd_must)
                emb_jd_ideal /= np.linalg.norm(emb_jd_ideal)
                emb_jd_core /= np.linalg.norm(emb_jd_core)

                # Phase C: Write temporary files to support structural scoring & reasoning
                status_container.markdown("📁 **Step 3/5:** Initializing heuristics & structural scorers...")
                progress_bar.progress(60)
                
                # Temporarily write uploaded candidates array to disk to satisfy the Scorer/Reasoner file contracts
                temp_data_path = Path("temp_candidates_sandbox.json")
                with open(temp_data_path, "w", encoding="utf-8") as f:
                    json.dump(candidates, f)

                # Temporarily write candidate_ids list
                temp_ids_path = Path("artifacts/candidate_ids.json")
                cids_list = [c["candidate_id"] for c in candidates]
                with open(temp_ids_path, "w", encoding="utf-8") as f:
                    json.dump(cids_list, f)

                # Compute inline integrity details and write integrity_flags.pkl temporarily
                temp_integ_data = {}
                for cand in candidates:
                    cid = cand["candidate_id"]
                    score, flags = compute_inline_integrity_score(cand)
                    temp_integ_data[cid] = {"score": score, "flags": flags}
                
                temp_integ_path = Path("artifacts/integrity_flags.pkl")
                with open(temp_integ_path, "wb") as f:
                    pickle.dump(temp_integ_data, f)

                # Instantiate Scorer and Reasoner classes using the temporary sandbox files
                struct_scorer = StructuralScorer(artifacts_dir="artifacts/", data_file=str(temp_data_path))
                reasoner = ReasoningGenerator(artifacts_dir="artifacts/", data_file=str(temp_data_path))

                # Phase D: Evaluate all candidates
                status_container.markdown("🧮 **Step 4/5:** Computing sub-scores and fusing metrics...")
                progress_bar.progress(85)
                
                fusion_results = []
                for cand in candidates:
                    cid = cand["candidate_id"]
                    
                    # 1. Semantic Score (on-the-fly)
                    # title + skills only capped to 150 chars (matching real embed.py)
                    profile = cand.get("profile", {})
                    skills_short = [s.get("name", "") for s in cand.get("skills", [])][:10]
                    cand_text = f"{profile.get('current_title', '')} | Skills: {', '.join(skills_short)}"[:150]
                    
                    emb_cand = model.encode(cand_text, convert_to_numpy=True)
                    emb_cand /= np.linalg.norm(emb_cand)
                    
                    sim_must = float(np.dot(emb_cand, emb_jd_must))
                    sim_ideal = float(np.dot(emb_cand, emb_jd_ideal))
                    sim_core = float(np.dot(emb_cand, emb_jd_core))
                    
                    w_semantic_sim = 0.50 * sim_must + 0.30 * sim_ideal + 0.20 * sim_core
                    score_semantic = float(max(0.0, min(1.0, w_semantic_sim)))

                    # 2. Structural Score (Direct class invocation)
                    score_structural = struct_scorer.score(cid)

                    # 3. Skill Graph Score (Keyword overlap approximation)
                    score_skill = compute_inline_skill_score(cand, must_have_kws, nice_to_have_kws)

                    # 4. Behavioral Score (Inline cluster calculation)
                    score_behavioral = compute_inline_behavioral_score(cand)

                    # 5. Integrity Score (Inline flag calculation)
                    score_integrity, integrity_flags = temp_integ_data[cid]["score"], temp_integ_data[cid]["flags"]

                    # Combined Fusion Score
                    combined_score = (
                        0.20 * score_semantic +
                        0.20 * score_skill +
                        0.15 * score_behavioral +
                        0.45 * score_structural
                    )
                    final_score = float(combined_score * score_integrity)
                    
                    # Save results along with subscores for reasoning generation
                    subscores = {
                        "semantic": score_semantic,
                        "skill_graph": score_skill,
                        "behavioral": score_behavioral,
                        "structural": score_structural,
                        "integrity": score_integrity
                    }
                    fusion_results.append({
                        "candidate_id": cid,
                        "score": final_score,
                        "subscores": subscores,
                        "profile": profile,
                        "flags": integrity_flags
                    })

                # Phase E: Ranking & Reasoning
                status_container.markdown("✍️ **Step 5/5:** Executing tie-breaker sorting and generating reasoning explanations...")
                progress_bar.progress(95)
                
                # Sort descending by score, tie-break by candidate_id ascending lexicographically
                fusion_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
                
                final_rows = []
                for idx, item in enumerate(fusion_results):
                    rank = idx + 1
                    cid = item["candidate_id"]
                    score = item["score"]
                    subscores = item["subscores"]
                    
                    # Call standard ReasoningGenerator
                    reasoning_str = reasoner.generate(cid, rank, score, subscores)
                    
                    final_rows.append({
                        "candidate_id": cid,
                        "rank": rank,
                        "score": f"{score:.6f}",
                        "name": item["profile"].get("anonymized_name", "Anonymized"),
                        "title": item["profile"].get("current_title", "N/A"),
                        "location": item["profile"].get("location", "N/A"),
                        "reasoning": reasoning_str
                    })

                # Completed!
                progress_bar.progress(100)
                status_container.success("🎉 ARIA Ranking Engine processing completed successfully!")
                
                # -----------------------------------------------------------
                # Section 3: Results Display
                # -----------------------------------------------------------
                df_results = pd.DataFrame(final_rows)
                
                st.subheader("🏆 Ranked Candidates (Top 100)")
                st.dataframe(df_results[["rank", "candidate_id", "name", "title", "location", "score", "reasoning"]], use_container_width=True)

                # Download Button for submission CSV
                csv_df = df_results[["candidate_id", "rank", "score", "reasoning"]]
                csv_content = csv_df.to_csv(index=False, encoding="utf-8")
                
                st.download_button(
                    label="📥 Download Ranked CSV Submission File",
                    data=csv_content,
                    file_name="submission_sandbox.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Score Monotonicity Visualization
                st.subheader("📈 Score Distribution (Rank Monotonicity)")
                chart_data = pd.DataFrame({
                    "Rank": df_results["rank"].astype(int),
                    "Score": df_results["score"].astype(float)
                })
                st.bar_chart(chart_data.set_index("Rank"), use_container_width=True)

            except Exception as e:
                st.error(f"Pipeline execution crashed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                # Cleanup sandbox temporary files to maintain clean repository layout
                for p in [temp_data_path, temp_ids_path, temp_integ_path]:
                    if p.exists():
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
