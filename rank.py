#!/usr/bin/env python3
"""
rank.py — Main execution entrypoint for the ARIA candidate ranking pipeline.
Consumes precomputed pipeline artifacts to score and rank candidates.

CLI Signature:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

from src.fusion import FusionEngine
from src.reasoning import ReasoningGenerator

def load_candidate_ids(candidates_path: Path) -> list[str]:
    """Streams candidate IDs from the target data file.
    
    Supports both JSON arrays (for testing) and streaming JSONL (for production).
    """
    candidate_ids = []
    
    if candidates_path.suffix.lower() == ".json":
        # Parse as standard JSON array
        with open(candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, item in enumerate(data):
            cid = item.get("candidate_id")
            if cid:
                candidate_ids.append(cid)
    else:
        # Stream line-by-line from JSONL
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        cand = json.loads(line)
                        cid = cand.get("candidate_id")
                        if cid:
                            candidate_ids.append(cid)
                    except json.JSONDecodeError as e:
                        print(f"[rank] Warning: JSON decode error at line {line_idx+1}: {e}", file=sys.stderr)
                        
    return candidate_ids

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the JD profile.")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (or sample_candidates.json)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write output submission.csv"
    )
    args = parser.parse_args()

    t_start = time.time()
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)

    if not candidates_path.exists():
        print(f"Error: Target candidates path does not exist: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[rank] Initializing ARIA Fusion Engine and Reasoning Generator ...", flush=True)
    try:
        # Initialize the FusionEngine, passing the data file path to scorers
        engine = FusionEngine(artifacts_dir="artifacts/", data_file=str(candidates_path))
        reasoner = ReasoningGenerator(artifacts_dir="artifacts/", data_file=str(candidates_path))
    except Exception as e:
        print(f"[rank] CRITICAL: Failed to initialize scorers/reasoner: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load the precomputed canonical IDs set to check for missing candidate IDs
    canonical_ids_path = Path("artifacts/candidate_ids.json")
    if not canonical_ids_path.exists():
        print(f"[rank] CRITICAL: Missing candidate_ids.json in artifacts directory.", file=sys.stderr)
        sys.exit(1)
    with open(canonical_ids_path, "r", encoding="utf-8") as f:
        canonical_set = set(json.load(f))

    print(f"[rank] Streaming candidate IDs from {candidates_path} ...", flush=True)
    all_cids = load_candidate_ids(candidates_path)
    print(f"[rank] Streamed {len(all_cids):,} candidate IDs.", flush=True)

    # Filter and warn on missing candidate IDs (skipping them to prevent crash)
    valid_cids = []
    skipped_count = 0
    for cid in all_cids:
        if cid in canonical_set:
            valid_cids.append(cid)
        else:
            print(f"[rank] WARNING: Candidate ID '{cid}' is not found in precomputed artifacts! Skipping candidate.", file=sys.stderr)
            skipped_count += 1

    if skipped_count > 0:
        print(f"[rank] Warning: Skipped {skipped_count} candidates due to missing precomputed artifacts.", flush=True)

    if not valid_cids:
        print(f"[rank] Error: No valid candidate IDs to score after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"[rank] Scoring {len(valid_cids):,} candidates ...", flush=True)
    try:
        scores = engine.score_all_candidates(valid_cids)
    except Exception as e:
        print(f"[rank] CRITICAL: Failed during candidate scoring: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Sort descending by score, tie-break by candidate_id ascending lexicographically
    # Python sorts are stable; sorting by a key of (-score, candidate_id) ensures both rules
    print(f"[rank] Ranking candidates ...", flush=True)
    ranked_candidates = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # Slice the top 100 candidates
    top_100 = ranked_candidates[:100]

    # Ensure out path directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output CSV in the exact required format: candidate_id,rank,score,reasoning
    print(f"[rank] Generating reasoning and writing top 100 candidates to {out_path} ...", flush=True)
    try:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Write required header row
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            # Write 100 data rows (ranks 1 to 100)
            for idx, (cid, score) in enumerate(top_100):
                rank = idx + 1
                
                # Fetch component subscores for reasoning context
                subscores = {
                    "semantic": engine.semantic.score(cid),
                    "skill_graph": engine.skill_graph.score(cid),
                    "behavioral": engine.behavioral.score(cid),
                    "structural": engine.structural.score(cid),
                    "integrity": engine.integrity.score(cid)
                }
                
                # Generate grounded reasoning string
                reasoning = reasoner.generate(cid, rank, score, subscores)
                
                # Format score to a high-precision float representation
                writer.writerow([cid, rank, f"{score:.6f}", reasoning])
    except Exception as e:
        print(f"[rank] CRITICAL: Failed writing submission CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    t_end = time.time()
    elapsed = t_end - t_start

    # ------------------------------------------------------------------
    # Honeypot Sanity Check
    # ------------------------------------------------------------------
    integrity_path = Path("artifacts/integrity_flags.pkl")
    integrity_data = {}
    if integrity_path.exists():
        with open(integrity_path, "rb") as f:
            integrity_data = pickle.load(f)

    print("\n" + "="*60, flush=True)
    print("HONEYPOT INTEGRITY SANITY CHECK REPORT", flush=True)
    print("="*60, flush=True)

    suspicious_count = 0
    flagged_candidates = []
    
    for rank_idx, (cid, score) in enumerate(top_100):
        record = integrity_data.get(cid, {})
        integ_score = record.get("score", 1.0) if isinstance(record, dict) else record
        flags = record.get("flags", []) if isinstance(record, dict) else []

        if integ_score < 0.5:
            suspicious_count += 1
            flagged_candidates.append((rank_idx + 1, cid, integ_score, flags))

    print(f"Total top 100 candidates with integrity score < 0.5: {suspicious_count} / 100", flush=True)
    if suspicious_count > 0:
        print("\nFlagged Suspicious Candidates:", flush=True)
        for rank, cid, integ_score, flags in flagged_candidates:
            print(f"  Rank {rank:3d} | ID: {cid} | Integrity Score: {integ_score:.4f} | Flags: {flags}", flush=True)

    if suspicious_count > 8:
        print("\n" + "!"*60, flush=True)
        print(f"WARNING: SUSPICIOUS CANDIDATE COUNT ({suspicious_count}) EXCEEDS THE ALLOWABLE HONEYPOT THRESHOLD (8%)!", flush=True)
        print("This submission is at high risk of disqualification under the 10% Honeypot rule.", flush=True)
        print("!"*60, flush=True)
    else:
        print("\nSUCCESS: Honeypot count is within safe limits (<= 8%).", flush=True)
    print("="*60 + "\n", flush=True)

    print(f"[rank] Pipeline execution completed successfully.", flush=True)
    print(f"[rank] Total wall-clock time: {elapsed:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
