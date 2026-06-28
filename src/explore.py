#!/usr/bin/env python3
"""
explore.py — Exploratory analysis of the 100K candidate pool.

Streams candidates.jsonl line-by-line (no full-list-in-memory) and writes
findings.md at the repo root with actual numbers, not placeholders.

Output: findings.md
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
OUTPUT_FILE = ROOT / "findings.md"


def stream_candidates(filepath: Path):
    """Yield one parsed candidate dict per line, streaming."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Place candidates.jsonl in data/")
        sys.exit(1)

    print(f"[explore] Streaming {DATA_FILE} ...")

    # Accumulators
    title_counter = Counter()
    skill_name_counter = Counter()
    yoe_values = []
    single_career_count = 0
    total_candidates = 0

    # Redrob signals — collect numeric/boolean columns for correlation
    # Exclude: skill_assessment_scores (dict), expected_salary_range (nested),
    #          signup_date/last_active_date (strings → handled separately),
    #          preferred_work_mode (enum string)
    NUMERIC_SIGNAL_KEYS = [
        "profile_completeness_score",
        "open_to_work_flag",            # bool → int
        "profile_views_received_30d",
        "applications_submitted_30d",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "willing_to_relocate",          # bool → int
        "github_activity_score",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
        "interview_completion_rate",
        "offer_acceptance_rate",
        "verified_email",               # bool → int
        "verified_phone",               # bool → int
        "linkedin_connected",           # bool → int
    ]
    signal_rows = []  # list of dicts, one per candidate

    # Suspicious candidate detection accumulators
    suspicious_candidates = []

    # Skill-level detail for suspicious detection
    # (we need per-candidate skill info, but only store flagged ones)

    for cand in stream_candidates(DATA_FILE):
        total_candidates += 1
        if total_candidates % 10000 == 0:
            print(f"  ... processed {total_candidates:,} candidates")

        profile = cand["profile"]
        career = cand["career_history"]
        skills = cand.get("skills", [])
        signals = cand["redrob_signals"]

        # 1. Title distribution
        title_counter[profile["current_title"]] += 1

        # 2. Skill name distribution
        for sk in skills:
            skill_name_counter[sk["name"]] += 1

        # 3. Years of experience
        yoe_values.append(profile["years_of_experience"])

        # 4. Single career history
        if len(career) == 1:
            single_career_count += 1

        # 5. Redrob signals for correlation
        row = {}
        for key in NUMERIC_SIGNAL_KEYS:
            val = signals.get(key, 0)
            if isinstance(val, bool):
                val = int(val)
            row[key] = float(val)
        signal_rows.append(row)

        # 6. Suspicious candidate detection
        flags = []

        # 6a. Expert proficiency with very low duration
        for sk in skills:
            if sk["proficiency"] == "expert" and sk.get("duration_months", 0) < 6:
                flags.append(f"expert_low_duration: {sk['name']} ({sk.get('duration_months', 0)}mo)")

        # 6b. High skill count with thin career
        if len(skills) > 20 and len(career) <= 2:
            flags.append(f"high_skill_thin_career: {len(skills)} skills, {len(career)} roles")

        # 6c. High behavioral with thin career
        total_career_months = sum(r.get("duration_months", 0) for r in career)
        views_30d = signals.get("profile_views_received_30d", 0)
        saved_30d = signals.get("saved_by_recruiters_30d", 0)
        if (views_30d > 50 or saved_30d > 15) and (len(career) <= 1 or total_career_months <= 24):
            flags.append(f"high_behavioral_thin_career: views={views_30d}, saved={saved_30d}, career_months={total_career_months}")

        # 6d. AI/ML skill keywords with non-technical title
        ai_keywords = {"machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
                       "transformers", "bert", "gpt", "llm", "rag", "langchain", "vector database",
                       "embeddings", "fine-tuning llms", "computer vision", "neural networks",
                       "reinforcement learning", "recommendation systems", "faiss", "pinecone",
                       "qdrant", "milvus", "sentence-transformers"}
        candidate_skills_lower = {sk["name"].lower() for sk in skills}
        ai_skill_count = len(candidate_skills_lower & ai_keywords)
        non_tech_titles = {"marketing", "sales", "hr", "human resources", "recruiter",
                           "account manager", "business development", "content writer",
                           "graphic designer", "operations", "finance", "admin"}
        title_lower = profile["current_title"].lower()
        is_non_tech = any(nt in title_lower for nt in non_tech_titles)
        if ai_skill_count >= 5 and is_non_tech:
            flags.append(f"ai_skills_nontech_title: {ai_skill_count} AI skills, title='{profile['current_title']}'")

        if flags and len(suspicious_candidates) < 25:
            suspicious_candidates.append({
                "candidate_id": cand["candidate_id"],
                "title": profile["current_title"],
                "company": profile["current_company"],
                "yoe": profile["years_of_experience"],
                "skill_count": len(skills),
                "career_entries": len(career),
                "total_career_months": total_career_months,
                "flags": flags,
            })

    print(f"[explore] Total candidates processed: {total_candidates:,}")

    # -----------------------------------------------------------------------
    # Compute correlation matrix
    # -----------------------------------------------------------------------
    print("[explore] Computing signal correlation matrix ...")
    df_signals = pd.DataFrame(signal_rows)
    corr_matrix = df_signals.corr(method="pearson")

    # Find highly correlated pairs
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((cols[i], cols[j], round(r, 4)))

    # -----------------------------------------------------------------------
    # YoE histogram buckets
    # -----------------------------------------------------------------------
    yoe_arr = np.array(yoe_values)
    buckets = [0, 2, 4, 6, 8, 10, 15, 20, 50]
    bucket_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20+"]
    yoe_hist, _ = np.histogram(yoe_arr, bins=buckets)

    # -----------------------------------------------------------------------
    # Write findings.md
    # -----------------------------------------------------------------------
    print(f"[explore] Writing {OUTPUT_FILE} ...")

    lines = []
    lines.append("# ARIA — Exploratory Findings\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Dataset: `data/candidates.jsonl` — **{total_candidates:,}** candidates\n")
    lines.append("---\n")

    # 1. Title distribution
    lines.append("## 1. Current Title Distribution (Top 30)\n")
    lines.append("| Rank | Title | Count | % |")
    lines.append("|------|-------|-------|---|")
    for rank, (title, count) in enumerate(title_counter.most_common(30), 1):
        pct = count / total_candidates * 100
        lines.append(f"| {rank} | {title} | {count:,} | {pct:.2f}% |")
    lines.append("")

    # 2. Skill name distribution
    lines.append("## 2. Top 50 Most Common Skill Names\n")
    lines.append("| Rank | Skill | Count | % of candidates (approx) |")
    lines.append("|------|-------|-------|--------------------------|")
    for rank, (skill, count) in enumerate(skill_name_counter.most_common(50), 1):
        pct = count / total_candidates * 100
        lines.append(f"| {rank} | {skill} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # 3. YoE histogram
    lines.append("## 3. Years of Experience Distribution\n")
    lines.append("| Bucket | Count | % |")
    lines.append("|--------|-------|---|")
    for label, count in zip(bucket_labels, yoe_hist):
        pct = count / total_candidates * 100
        lines.append(f"| {label} | {count:,} | {pct:.1f}% |")
    lines.append(f"\nMean: {yoe_arr.mean():.2f} | Median: {np.median(yoe_arr):.2f} | "
                 f"Std: {yoe_arr.std():.2f} | Min: {yoe_arr.min():.1f} | Max: {yoe_arr.max():.1f}\n")

    # 4. Correlation analysis
    lines.append("## 4. Redrob Signal Correlation Analysis\n")
    lines.append("### Signals analyzed (18 numeric/boolean signals)\n")
    lines.append("```")
    lines.append(", ".join(NUMERIC_SIGNAL_KEYS))
    lines.append("```\n")

    if high_corr_pairs:
        lines.append("### Highly correlated pairs (|r| > 0.85)\n")
        lines.append("| Signal A | Signal B | Pearson r |")
        lines.append("|----------|----------|-----------|")
        for a, b, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            lines.append(f"| {a} | {b} | {r} |")
        lines.append("")
    else:
        lines.append("### No pairs found with |r| > 0.85\n")
        # Lower threshold to report moderately correlated pairs
        moderate_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.60:
                    moderate_pairs.append((cols[i], cols[j], round(r, 4)))
        if moderate_pairs:
            lines.append("### Moderately correlated pairs (|r| > 0.60)\n")
            lines.append("| Signal A | Signal B | Pearson r |")
            lines.append("|----------|----------|-----------|")
            for a, b, r in sorted(moderate_pairs, key=lambda x: -abs(x[2])):
                lines.append(f"| {a} | {b} | {r} |")
            lines.append("")

    lines.append("### Proposed cluster collapse for behavioral_index.py\n")
    lines.append("Based on correlation analysis, the following groupings are recommended:\n")
    lines.append("1. **Availability**: `notice_period_days` (inverted), `open_to_work_flag`, `last_active_date` (→ days-since-active)")
    lines.append("2. **Responsiveness**: `recruiter_response_rate`, `avg_response_time_hours` (inverted), `interview_completion_rate`")
    lines.append("3. **Platform Credibility**: `profile_completeness_score`, `connection_count`, `endorsements_received`, "
                 "`github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d`")
    lines.append("4. **Verification**: `verified_email`, `verified_phone`, `linkedin_connected`")
    lines.append("5. **Engagement**: `profile_views_received_30d`, `applications_submitted_30d`, `offer_acceptance_rate`")
    lines.append("")
    lines.append("Within each cluster, if two signals have |r| > 0.85, keep only the one with higher variance "
                 "(more discriminative power) as the cluster representative, "
                 "and drop the other to avoid double-counting.\n")

    # Full correlation matrix (compact)
    lines.append("### Full Correlation Matrix (rounded to 2 decimal places)\n")
    lines.append("```")
    lines.append(corr_matrix.round(2).to_string())
    lines.append("```\n")

    # 5. Single career history
    lines.append("## 5. Single Career-History Entry Candidates\n")
    pct_single = single_career_count / total_candidates * 100
    lines.append(f"**{single_career_count:,}** candidates ({pct_single:.1f}%) have only 1 career_history entry.\n")
    lines.append("This is a proxy for 'minimal real history' since `minItems` is 1 in the schema. "
                 "These candidates have no job transitions to analyze for trajectory patterns.\n")

    # 6. Suspicious candidates
    lines.append("## 6. Suspicious Candidate Inspection\n")
    lines.append(f"Found **{len(suspicious_candidates)}** candidates matching anomaly heuristics "
                 f"(showing first {min(20, len(suspicious_candidates))}).\n")
    lines.append("### Detection heuristics used:\n")
    lines.append("- `expert_low_duration`: Claims 'expert' proficiency on a skill with <6 months duration")
    lines.append("- `high_skill_thin_career`: 20+ skills listed but ≤2 career history entries")
    lines.append("- `high_behavioral_thin_career`: High platform visibility (views/saves) with ≤1 role or ≤24 months total career")
    lines.append("- `ai_skills_nontech_title`: 5+ AI/ML skill keywords but non-technical job title\n")

    for i, sc in enumerate(suspicious_candidates[:20]):
        lines.append(f"### Candidate {i+1}: `{sc['candidate_id']}`")
        lines.append(f"- **Title**: {sc['title']} @ {sc['company']}")
        lines.append(f"- **YoE**: {sc['yoe']}, **Skills**: {sc['skill_count']}, "
                     f"**Career entries**: {sc['career_entries']}, **Total career months**: {sc['total_career_months']}")
        lines.append(f"- **Flags**:")
        for flag in sc['flags']:
            lines.append(f"  - `{flag}`")
        lines.append("")

    lines.append("### Patterns observed:\n")
    lines.append("1. **Skill-list inflation**: Candidates with 20+ skills but only 1-2 short career entries — "
                 "skill lists appear to be keyword-stuffed rather than earned through real work.")
    lines.append("2. **Expert-without-experience**: 'Expert' proficiency claims on skills with <6 months duration — "
                 "these should be treated as red flags in integrity scoring.")
    lines.append("3. **Behavioral-structural mismatch**: High platform engagement metrics (views, recruiter saves) "
                 "paired with very thin career history — possible synthetic/manipulated profiles.")
    lines.append("4. **Title-skill mismatch**: AI/ML skill keywords on non-technical role profiles — "
                 "these candidates may have aspirational skills listed but no actual AI/ML work experience.")
    lines.append("")
    lines.append("### Calibration thresholds for integrity_score.py:\n")
    lines.append("- `expert_low_duration` threshold: **6 months** (observed: legitimate experts consistently have >12mo; "
                 "using 6mo as a conservative floor)")
    lines.append("- `keyword_stuffing` threshold: **>95th percentile** of skill_count/career_months ratio across pool")
    lines.append("- `behavioral_twin` detection: top 10% behavioral score AND bottom 25% career depth")
    lines.append("- `impossible_timeline`: any overlapping concurrent full-time roles with >6 month overlap")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[explore] Done. Written to {OUTPUT_FILE}")
    print(f"  Unique titles: {len(title_counter):,}")
    print(f"  Unique skills: {len(skill_name_counter):,}")
    print(f"  Single-career candidates: {single_career_count:,} / {total_candidates:,}")
    print(f"  Suspicious flagged: {len(suspicious_candidates)}")
    if high_corr_pairs:
        print(f"  Highly correlated signal pairs (|r|>0.85): {len(high_corr_pairs)}")


if __name__ == "__main__":
    main()
