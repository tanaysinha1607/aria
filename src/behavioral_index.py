#!/usr/bin/env python3
"""
behavioral_index.py — Behavioral signal clustering & normalization.

Collapses the 23 raw redrob_signals into 5 cluster representatives,
each normalized to [0,1]. Uses percentile-rank normalization (robust
to outliers) and incorporates findings from explore.py's correlation
analysis to avoid double-counting correlated signals.

Key design choice: last_active_date is converted to "days since last active"
as a numeric feature for the Availability cluster — the correlation matrix
in explore.py excludes it since it's a raw date string, but it carries
critical signal about candidate reachability.

Clusters:
  0: Availability  — notice_period (inverted), open_to_work, days_since_active (inverted)
  1: Responsiveness — recruiter_response_rate, avg_response_time (inverted), interview_completion_rate
  2: Platform Credibility — profile_completeness, connection_count, endorsements, github_activity,
                            search_appearance_30d, saved_by_recruiters_30d
  3: Verification   — verified_email, verified_phone, linkedin_connected
  4: Engagement     — profile_views_30d, applications_submitted_30d, offer_acceptance_rate

Artifacts produced:
  - artifacts/behavioral_features.npy       (100K × 5, float32)
  - artifacts/behavioral_features_meta.json (column names + normalization details)
"""

import json
import sys
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
CANDIDATE_IDS_FILE = ARTIFACTS_DIR / "candidate_ids.json"

OUT_FEATURES = ARTIFACTS_DIR / "behavioral_features.npy"
OUT_META = ARTIFACTS_DIR / "behavioral_features_meta.json"

# Reference date for "days since" calculation
REFERENCE_DATE = date(2026, 6, 27)  # today's date as reference point


def parse_date_to_days_since(date_str: str) -> float:
    """Convert a date string to days-since-reference. Higher = more stale."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        delta = (REFERENCE_DATE - d).days
        return max(0.0, float(delta))
    except (ValueError, TypeError):
        return 365.0  # default to ~1 year stale if unparseable


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile-rank normalize an array to [0, 1].

    Handles NaN/inf by replacing with 0 before ranking.
    Uses scipy.stats.rankdata with 'average' method.
    """
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    ranks = scipy_stats.rankdata(clean, method="average")
    # Normalize to [0, 1]
    n = len(ranks)
    if n <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    return ((ranks - 1) / (n - 1)).astype(np.float32)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load canonical candidate ordering
    # ------------------------------------------------------------------
    if CANDIDATE_IDS_FILE.exists():
        with open(CANDIDATE_IDS_FILE, "r", encoding="utf-8") as f:
            canonical_ids = json.load(f)
        print(f"[behavioral] Loaded canonical ordering: {len(canonical_ids):,} candidates")
    else:
        canonical_ids = None
        print("[behavioral] WARNING: candidate_ids.json not found — will use data file order")

    # ------------------------------------------------------------------
    # Stream and extract raw signals
    # ------------------------------------------------------------------
    print(f"[behavioral] Extracting raw signals from {DATA_FILE} ...")

    # Raw signal arrays — will be populated in data-file order, then reordered
    raw_data = {
        # Availability cluster
        "notice_period_days": [],
        "open_to_work_flag": [],
        "days_since_active": [],       # DERIVED from last_active_date

        # Responsiveness cluster
        "recruiter_response_rate": [],
        "avg_response_time_hours": [],
        "interview_completion_rate": [],

        # Platform Credibility cluster
        "profile_completeness_score": [],
        "connection_count": [],
        "endorsements_received": [],
        "github_activity_score": [],
        "search_appearance_30d": [],
        "saved_by_recruiters_30d": [],

        # Verification cluster
        "verified_email": [],
        "verified_phone": [],
        "linkedin_connected": [],

        # Engagement cluster
        "profile_views_received_30d": [],
        "applications_submitted_30d": [],
        "offer_acceptance_rate": [],
    }

    data_order_ids = []
    total = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            signals = cand["redrob_signals"]
            total += 1

            data_order_ids.append(cand["candidate_id"])

            # Extract raw values
            raw_data["notice_period_days"].append(float(signals.get("notice_period_days", 90)))
            raw_data["open_to_work_flag"].append(1.0 if signals.get("open_to_work_flag", False) else 0.0)
            raw_data["days_since_active"].append(
                parse_date_to_days_since(signals.get("last_active_date", ""))
            )
            raw_data["recruiter_response_rate"].append(float(signals.get("recruiter_response_rate", 0.0)))
            raw_data["avg_response_time_hours"].append(float(signals.get("avg_response_time_hours", 48.0)))
            raw_data["interview_completion_rate"].append(float(signals.get("interview_completion_rate", 0.0)))
            raw_data["profile_completeness_score"].append(float(signals.get("profile_completeness_score", 0)))
            raw_data["connection_count"].append(float(signals.get("connection_count", 0)))
            raw_data["endorsements_received"].append(float(signals.get("endorsements_received", 0)))

            # github_activity_score: -1 means no GitHub — treat as 0
            github = float(signals.get("github_activity_score", -1))
            raw_data["github_activity_score"].append(max(0.0, github))

            raw_data["search_appearance_30d"].append(float(signals.get("search_appearance_30d", 0)))
            raw_data["saved_by_recruiters_30d"].append(float(signals.get("saved_by_recruiters_30d", 0)))

            raw_data["verified_email"].append(1.0 if signals.get("verified_email", False) else 0.0)
            raw_data["verified_phone"].append(1.0 if signals.get("verified_phone", False) else 0.0)
            raw_data["linkedin_connected"].append(1.0 if signals.get("linkedin_connected", False) else 0.0)

            raw_data["profile_views_received_30d"].append(float(signals.get("profile_views_received_30d", 0)))
            raw_data["applications_submitted_30d"].append(float(signals.get("applications_submitted_30d", 0)))

            # offer_acceptance_rate: -1 means no offer history — treat as 0.5 (neutral)
            oar = float(signals.get("offer_acceptance_rate", -1))
            raw_data["offer_acceptance_rate"].append(0.5 if oar < 0 else oar)

            if total % 10000 == 0:
                print(f"  ... {total:,} candidates extracted")

    print(f"[behavioral] Extracted {total:,} candidates")

    # Convert to numpy arrays
    for key in raw_data:
        raw_data[key] = np.array(raw_data[key], dtype=np.float32)

    # ------------------------------------------------------------------
    # Build cluster scores
    # ------------------------------------------------------------------
    print("[behavioral] Computing cluster scores ...")

    N = total

    # Cluster 0: Availability
    # notice_period: LOWER is better → invert
    notice_norm = 1.0 - percentile_rank(raw_data["notice_period_days"])
    otw_norm = raw_data["open_to_work_flag"]  # already 0/1
    # days_since_active: LOWER is better (more recent) → invert
    active_norm = 1.0 - percentile_rank(raw_data["days_since_active"])
    availability = (0.3 * notice_norm + 0.35 * otw_norm + 0.35 * active_norm).astype(np.float32)

    # Cluster 1: Responsiveness
    resp_rate_norm = percentile_rank(raw_data["recruiter_response_rate"])
    # avg_response_time: LOWER is better → invert
    resp_time_norm = 1.0 - percentile_rank(raw_data["avg_response_time_hours"])
    interview_norm = percentile_rank(raw_data["interview_completion_rate"])
    responsiveness = ((resp_rate_norm + resp_time_norm + interview_norm) / 3.0).astype(np.float32)

    # Cluster 2: Platform Credibility
    completeness_norm = percentile_rank(raw_data["profile_completeness_score"])
    connection_norm = percentile_rank(raw_data["connection_count"])
    endorsement_norm = percentile_rank(raw_data["endorsements_received"])
    github_norm = percentile_rank(raw_data["github_activity_score"])
    search_norm = percentile_rank(raw_data["search_appearance_30d"])
    saved_norm = percentile_rank(raw_data["saved_by_recruiters_30d"])
    credibility = ((completeness_norm + connection_norm + endorsement_norm +
                    github_norm + search_norm + saved_norm) / 6.0).astype(np.float32)

    # Cluster 3: Verification
    verification = ((raw_data["verified_email"] + raw_data["verified_phone"] +
                     raw_data["linkedin_connected"]) / 3.0).astype(np.float32)

    # Cluster 4: Engagement
    views_norm = percentile_rank(raw_data["profile_views_received_30d"])
    apps_norm = percentile_rank(raw_data["applications_submitted_30d"])
    oar_norm = percentile_rank(raw_data["offer_acceptance_rate"])
    engagement = ((views_norm + apps_norm + oar_norm) / 3.0).astype(np.float32)

    # Stack into (N, 5) matrix
    features_data_order = np.column_stack([
        availability, responsiveness, credibility, verification, engagement
    ]).astype(np.float32)

    print(f"  Features shape (data order): {features_data_order.shape}")

    # ------------------------------------------------------------------
    # Reorder to canonical candidate_ids.json order
    # ------------------------------------------------------------------
    if canonical_ids is not None:
        print("[behavioral] Reordering to canonical candidate_ids.json order ...")
        id_to_data_idx = {cid: i for i, cid in enumerate(data_order_ids)}
        features = np.zeros((len(canonical_ids), 5), dtype=np.float32)
        for out_idx, cid in enumerate(canonical_ids):
            data_idx = id_to_data_idx.get(cid)
            if data_idx is not None:
                features[out_idx] = features_data_order[data_idx]
            else:
                print(f"  WARNING: {cid} not found in data — zeroed out")
    else:
        features = features_data_order

    # Clamp to [0, 1]
    features = np.clip(features, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.save(OUT_FEATURES, features)

    meta = {
        "columns": [
            "availability",
            "responsiveness",
            "platform_credibility",
            "verification",
            "engagement"
        ],
        "column_details": {
            "availability": {
                "signals": ["notice_period_days (inverted)", "open_to_work_flag",
                            "days_since_active (derived from last_active_date, inverted)"],
                "weights": [0.3, 0.35, 0.35],
                "normalization": "percentile_rank for continuous, raw for binary"
            },
            "responsiveness": {
                "signals": ["recruiter_response_rate", "avg_response_time_hours (inverted)",
                            "interview_completion_rate"],
                "weights": "equal (1/3 each)",
                "normalization": "percentile_rank"
            },
            "platform_credibility": {
                "signals": ["profile_completeness_score", "connection_count",
                            "endorsements_received", "github_activity_score (-1→0)",
                            "search_appearance_30d", "saved_by_recruiters_30d"],
                "weights": "equal (1/6 each)",
                "normalization": "percentile_rank"
            },
            "verification": {
                "signals": ["verified_email", "verified_phone", "linkedin_connected"],
                "weights": "equal (1/3 each)",
                "normalization": "raw binary (0 or 1)"
            },
            "engagement": {
                "signals": ["profile_views_received_30d", "applications_submitted_30d",
                            "offer_acceptance_rate (-1→0.5)"],
                "weights": "equal (1/3 each)",
                "normalization": "percentile_rank"
            }
        },
        "row_order": "matches candidate_ids.json",
        "reference_date_for_days_since": str(REFERENCE_DATE),
        "total_candidates": len(features),
        "shape": list(features.shape),
        "dtype": "float32",
        "value_range": "[0.0, 1.0]"
    }

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    t_total = time.time() - t0
    print(f"\n[behavioral] COMPLETE in {t_total:.1f}s")
    print(f"  Output: {OUT_FEATURES} — shape={features.shape}, dtype={features.dtype}")
    print(f"  Meta: {OUT_META}")
    print(f"  Column stats:")
    for i, name in enumerate(meta["columns"]):
        col = features[:, i]
        print(f"    {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"min={col.min():.4f}, max={col.max():.4f}")


if __name__ == "__main__":
    main()
