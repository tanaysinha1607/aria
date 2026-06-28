#!/usr/bin/env python3
"""
integrity_score.py — Honeypot/anomaly pre-computation.

Computes per-candidate integrity flags and a composite score in [0,1]
(1.0 = clean, descending toward 0 = more flags triggered).

This is a SOFT penalty multiplier, NOT a hard filter — the downstream
ranker (Atharv's half) decides how to apply it.

Flags:
  - skill_experience_mismatch: "expert" proficiency with <6mo duration
  - keyword_stuffing: abnormally high skill-count-to-career-depth ratio
  - behavioral_twin: high behavioral signals + thin career history
  - impossible_timeline: overlapping concurrent full-time roles

Artifacts produced:
  - artifacts/integrity_flags.pkl — dict[candidate_id → {"score": float, "flags": [str]}]
"""

import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
CANDIDATE_IDS_FILE = ARTIFACTS_DIR / "candidate_ids.json"
BEHAVIORAL_FILE = ARTIFACTS_DIR / "behavioral_features.npy"

OUT_INTEGRITY = ARTIFACTS_DIR / "integrity_flags.pkl"

# ---------------------------------------------------------------------------
# Thresholds (calibrated from explore.py findings)
# ---------------------------------------------------------------------------
EXPERT_LOW_DURATION_MONTHS = 6      # "expert" with < this many months → flag
SKILL_STUFFING_PERCENTILE = 95      # top 5% of skill_count/career_months ratio
HIGH_BEHAVIORAL_PERCENTILE = 90     # top 10% of behavioral cluster mean
LOW_CAREER_DEPTH_PERCENTILE = 25    # bottom 25% of total career months
OVERLAP_MONTHS_THRESHOLD = 6        # concurrent roles overlapping > this → flag

# Penalty magnitudes
PENALTY_EXPERT_MISMATCH_PER = 0.15  # per flagged skill
PENALTY_EXPERT_MISMATCH_CAP = 0.45  # max total
PENALTY_KEYWORD_STUFFING = 0.25
PENALTY_BEHAVIORAL_TWIN = 0.20
PENALTY_IMPOSSIBLE_TIMELINE_PER = 0.30
PENALTY_IMPOSSIBLE_TIMELINE_CAP = 0.60


def parse_date(d: str) -> datetime | None:
    """Parse a date string, return None on failure."""
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def check_impossible_timeline(career_history: list) -> list:
    """Check for overlapping concurrent full-time roles.

    Returns list of overlap descriptions.
    """
    overlaps = []
    # Filter to roles with valid date ranges
    dated_roles = []
    for role in career_history:
        start = parse_date(role.get("start_date", ""))
        end = parse_date(role.get("end_date", ""))
        if start is None:
            continue
        # For current roles, use today as end
        if end is None:
            end = datetime(2026, 6, 27)
        dated_roles.append({
            "title": role.get("title", "Unknown"),
            "company": role.get("company", "Unknown"),
            "start": start,
            "end": end,
            "duration": role.get("duration_months", 0),
        })

    # Check all pairs for overlap
    for i in range(len(dated_roles)):
        for j in range(i + 1, len(dated_roles)):
            r1 = dated_roles[i]
            r2 = dated_roles[j]

            # Overlap = max(0, min(end1, end2) - max(start1, start2))
            overlap_start = max(r1["start"], r2["start"])
            overlap_end = min(r1["end"], r2["end"])

            if overlap_end > overlap_start:
                overlap_days = (overlap_end - overlap_start).days
                overlap_months = overlap_days / 30.44

                if overlap_months > OVERLAP_MONTHS_THRESHOLD:
                    overlaps.append(
                        f"'{r1['title']}@{r1['company']}' overlaps with "
                        f"'{r2['title']}@{r2['company']}' by {overlap_months:.0f} months"
                    )

    return overlaps


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Pass 1: Collect pool-wide statistics for threshold calibration
    # ------------------------------------------------------------------
    print(f"[integrity] Pass 1: Collecting pool-wide statistics from {DATA_FILE} ...")

    skill_to_career_ratios = []  # skill_count / max(total_career_months, 1)
    career_month_totals = []
    candidate_data_cache = []    # we need a second pass, so cache key fields

    total = 0
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total += 1

            skills = cand.get("skills", [])
            career = cand.get("career_history", [])
            total_career_months = sum(r.get("duration_months", 0) for r in career)

            # Cache for pass 2 (only what we need — minimize memory)
            candidate_data_cache.append({
                "candidate_id": cand["candidate_id"],
                "skills": [(sk["name"], sk.get("proficiency", ""), sk.get("duration_months", 0))
                           for sk in skills],
                "career_entry_count": len(career),
                "total_career_months": total_career_months,
                "career_history": career,  # needed for timeline check
            })

            ratio = len(skills) / max(total_career_months, 1)
            skill_to_career_ratios.append(ratio)
            career_month_totals.append(total_career_months)

            if total % 10000 == 0:
                print(f"  ... {total:,} candidates scanned")

    print(f"[integrity] Pass 1 complete: {total:,} candidates")

    # Compute percentile thresholds
    ratios_arr = np.array(skill_to_career_ratios)
    career_arr = np.array(career_month_totals)

    stuffing_threshold = np.percentile(ratios_arr, SKILL_STUFFING_PERCENTILE)
    low_career_threshold = np.percentile(career_arr, LOW_CAREER_DEPTH_PERCENTILE)

    print(f"  Keyword stuffing threshold (P{SKILL_STUFFING_PERCENTILE}): "
          f"{stuffing_threshold:.4f} skills/month")
    print(f"  Low career depth threshold (P{LOW_CAREER_DEPTH_PERCENTILE}): "
          f"{low_career_threshold:.1f} months")

    # Load behavioral features for twin detection
    behavioral_features = None
    behavioral_means = None
    if BEHAVIORAL_FILE.exists():
        behavioral_features = np.load(BEHAVIORAL_FILE)
        behavioral_means = behavioral_features.mean(axis=1)  # average across clusters
        high_behavioral_threshold = np.percentile(behavioral_means, HIGH_BEHAVIORAL_PERCENTILE)
        print(f"  High behavioral threshold (P{HIGH_BEHAVIORAL_PERCENTILE}): "
              f"{high_behavioral_threshold:.4f}")
    else:
        print("  WARNING: behavioral_features.npy not found — skipping twin detection")
        high_behavioral_threshold = 1.0  # effectively disables

    # Load canonical ordering
    canonical_ids = None
    id_to_canonical_idx = {}
    if CANDIDATE_IDS_FILE.exists():
        with open(CANDIDATE_IDS_FILE, "r", encoding="utf-8") as f:
            canonical_ids = json.load(f)
        id_to_canonical_idx = {cid: i for i, cid in enumerate(canonical_ids)}
        print(f"  Canonical ordering loaded: {len(canonical_ids):,} candidates")

    # ------------------------------------------------------------------
    # Pass 2: Score each candidate
    # ------------------------------------------------------------------
    print("[integrity] Pass 2: Computing integrity scores ...")

    results = {}
    flag_counts = Counter()

    for data_idx, cached in enumerate(candidate_data_cache):
        cid = cached["candidate_id"]
        flags = []
        penalty = 0.0

        # --- Flag 1: Skill-experience mismatch ---
        expert_mismatch_count = 0
        for skill_name, proficiency, duration in cached["skills"]:
            if proficiency == "expert" and duration < EXPERT_LOW_DURATION_MONTHS:
                expert_mismatch_count += 1
                if expert_mismatch_count <= 3:  # only detail first 3
                    flags.append(
                        f"skill_experience_mismatch: '{skill_name}' expert with {duration}mo"
                    )
        if expert_mismatch_count > 3:
            flags.append(f"skill_experience_mismatch: +{expert_mismatch_count - 3} more")
        penalty += min(expert_mismatch_count * PENALTY_EXPERT_MISMATCH_PER,
                       PENALTY_EXPERT_MISMATCH_CAP)
        if expert_mismatch_count > 0:
            flag_counts["skill_experience_mismatch"] += 1

        # --- Flag 2: Keyword stuffing ---
        ratio = len(cached["skills"]) / max(cached["total_career_months"], 1)
        is_stuffed = (ratio > stuffing_threshold) or \
                     (len(cached["skills"]) > 25 and cached["career_entry_count"] <= 2)
        if is_stuffed:
            flags.append(
                f"keyword_stuffing: {len(cached['skills'])} skills, "
                f"{cached['total_career_months']}mo career, ratio={ratio:.3f}"
            )
            penalty += PENALTY_KEYWORD_STUFFING
            flag_counts["keyword_stuffing"] += 1

        # --- Flag 3: Behavioral twin ---
        if behavioral_means is not None:
            canonical_idx = id_to_canonical_idx.get(cid, data_idx)
            if canonical_idx < len(behavioral_means):
                beh_score = behavioral_means[canonical_idx]
                is_twin = (beh_score > high_behavioral_threshold and
                           cached["total_career_months"] < low_career_threshold)
                if is_twin:
                    flags.append(
                        f"behavioral_twin: behavioral_mean={beh_score:.3f}, "
                        f"career_months={cached['total_career_months']}"
                    )
                    penalty += PENALTY_BEHAVIORAL_TWIN
                    flag_counts["behavioral_twin"] += 1

        # --- Flag 4: Impossible timeline ---
        overlaps = check_impossible_timeline(cached["career_history"])
        for overlap_desc in overlaps:
            flags.append(f"impossible_timeline: {overlap_desc}")
        timeline_penalty = min(len(overlaps) * PENALTY_IMPOSSIBLE_TIMELINE_PER,
                               PENALTY_IMPOSSIBLE_TIMELINE_CAP)
        penalty += timeline_penalty
        if overlaps:
            flag_counts["impossible_timeline"] += 1

        # --- Compute final score ---
        score = max(0.0, 1.0 - penalty)

        results[cid] = {
            "score": round(score, 4),
            "flags": flags,
        }

        if (data_idx + 1) % 10000 == 0:
            print(f"  ... {data_idx + 1:,}/{total:,} candidates scored")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(OUT_INTEGRITY, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    t_total = time.time() - t0

    # Stats
    scores = np.array([v["score"] for v in results.values()])
    flagged = sum(1 for v in results.values() if v["flags"])

    print(f"\n[integrity] COMPLETE in {t_total:.1f}s")
    print(f"  Total candidates: {len(results):,}")
    print(f"  Flagged candidates: {flagged:,} ({flagged/len(results)*100:.1f}%)")
    print(f"  Score distribution:")
    print(f"    Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print(f"    Min: {scores.min():.4f}, Max: {scores.max():.4f}")
    print(f"    Score=1.0 (fully clean): {np.sum(scores == 1.0):,}")
    print(f"    Score<0.5 (heavily flagged): {np.sum(scores < 0.5):,}")
    print(f"    Score=0.0 (max penalty): {np.sum(scores == 0.0):,}")
    print(f"  Flag type counts:")
    for flag_type, count in flag_counts.most_common():
        print(f"    {flag_type}: {count:,} candidates")
    print(f"  Output: {OUT_INTEGRITY}")


if __name__ == "__main__":
    main()
