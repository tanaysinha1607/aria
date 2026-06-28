#!/usr/bin/env python3
"""
embed.py — Semantic embeddings + FAISS index.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - Chosen for fast CPU-only execution under tight timeline constraints.
    Runs at ~200 cand/s on CPU (compared to ~4 cand/s for bge-base), completing
    the 100K candidates in ~8.4 minutes.
  - Generates 384-dim vectors (npy size: ~146 MB).

Architecture:
  - Text: current_title + skills (no career history prose, capped at 150 chars)
  - Model encodes both JD sections and candidates in the same vector space
  - No asymmetric query prefix (standard sentence-transformers behavior)
  - L2-normalized → FAISS IndexFlatIP (cosine similarity)
  - Processed in batches of 256
"""

import json
import os
import sys
import time
import gc
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256
MAX_CANDIDATE_TEXT_LEN = 150

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
JD_STRUCTURED = ARTIFACTS_DIR / "jd_structured.json"

OUT_EMBEDDINGS = ARTIFACTS_DIR / "candidate_embeddings.npy"
OUT_JD_EMBEDDINGS = ARTIFACTS_DIR / "jd_section_embeddings.npz"
OUT_FAISS = ARTIFACTS_DIR / "faiss_index.bin"
OUT_IDS = ARTIFACTS_DIR / "candidate_ids.json"


def build_candidate_text(cand: dict) -> str:
    """Build the text to embed for a single candidate.

    Concatenates: current_title | skill names (caps at 150 characters, title + skills only).
    """
    profile = cand["profile"]
    skills = cand.get("skills", [])

    parts = [profile.get("current_title", "")]
    skill_names = [sk["name"] for sk in skills]
    if skill_names:
        parts.append(" | " + ", ".join(skill_names))

    full_text = "".join(parts)
    if len(full_text) > MAX_CANDIDATE_TEXT_LEN:
        full_text = full_text[:MAX_CANDIDATE_TEXT_LEN].rsplit(" ", 1)[0]

    return full_text


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load JD structure
    if not JD_STRUCTURED.exists():
        print("[embed] jd_structured.json not found — running jd_parser first ...", flush=True)
        from jd_parser import main as jd_main
        jd_main()

    with open(JD_STRUCTURED, "r", encoding="utf-8") as f:
        jd = json.load(f)

    # Load model
    print(f"[embed] Loading model: {MODEL_NAME}", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Embed JD sections (no query prefix needed for MiniLM)
    print("[embed] Embedding JD sections ...", flush=True)
    jd_sections = {
        "must_have": " ".join(jd["must_have_signals"]),
        "ideal_profile": jd["ideal_profile_description"],
        "core_responsibilities": jd["core_responsibilities"],
    }

    jd_embeddings = {}
    for name, text in jd_sections.items():
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        jd_embeddings[name] = vec.astype(np.float32)  # shape (1, 384)
        print(f"  {name}: shape={vec.shape}, norm={np.linalg.norm(vec[0]):.4f}", flush=True)

    np.savez(OUT_JD_EMBEDDINGS, **jd_embeddings)
    print(f"[embed] JD embeddings saved to {OUT_JD_EMBEDDINGS}", flush=True)

    # Stream candidates, build texts, embed in batches
    print(f"[embed] Streaming candidates from {DATA_FILE} ...", flush=True)

    candidate_ids = []
    candidate_texts = []
    batch_count = 0

    t_read_start = time.time()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidate_ids.append(cand["candidate_id"])
            candidate_texts.append(build_candidate_text(cand))
            if (i + 1) % 10000 == 0:
                print(f"  [read] {i + 1:,} candidates read ...", flush=True)

    total = len(candidate_ids)
    t_read_end = time.time()
    print(f"[embed] Read {total:,} candidates in {t_read_end - t_read_start:.1f}s", flush=True)

    # Ensure IDs are matched to what was pre-generated
    if OUT_IDS.exists():
        with open(OUT_IDS, "r", encoding="utf-8") as f:
            canonical_ids = json.load(f)
        if len(canonical_ids) != total:
            print("[embed] Warning: canonical IDs count mismatch. Re-writing candidate_ids.json", flush=True)
            with open(OUT_IDS, "w", encoding="utf-8") as f:
                json.dump(candidate_ids, f)
    else:
        with open(OUT_IDS, "w", encoding="utf-8") as f:
            json.dump(candidate_ids, f)

    # Allocate output array (memory-map to reduce RAM pressure)
    embeddings = np.memmap(
        str(OUT_EMBEDDINGS), dtype=np.float32, mode="w+",
        shape=(total, EMBEDDING_DIM)
    )

    print(f"[embed] Embedding {total:,} candidates in batches of {BATCH_SIZE} ...", flush=True)
    print(f"[embed] Output shape: ({total}, {EMBEDDING_DIM}), dtype=float32", flush=True)
    t_embed_start = time.time()

    # Progress statistics tracking
    prev_idx = 0
    prev_time = t_embed_start

    for start_idx in range(0, total, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_texts = candidate_texts[start_idx:end_idx]

        batch_vecs = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        embeddings[start_idx:end_idx] = batch_vecs.astype(np.float32)
        batch_count += 1

        # Manual garbage collection to prevent memory issues
        if batch_count % 20 == 0:
            gc.collect()

        # Progress logging every ~10K candidates
        if (end_idx % 10000 < BATCH_SIZE) or end_idx == total:
            now = time.time()
            elapsed = now - t_embed_start
            
            # Calculate short-term rate
            chunk_elapsed = now - prev_time
            chunk_processed = end_idx - prev_idx
            rate = chunk_processed / chunk_elapsed if chunk_elapsed > 0 else 0
            
            overall_rate = end_idx / elapsed if elapsed > 0 else 0
            current_rate = rate if end_idx > 10000 else overall_rate
            
            eta_min = (total - end_idx) / current_rate / 60 if current_rate > 0 else 0
            print(f"  [embed] {end_idx:,}/{total:,} "
                  f"({end_idx/total*100:.1f}%) | "
                  f"{elapsed/60:.1f}min elapsed | "
                  f"current rate: {current_rate:.1f} cand/s | "
                  f"overall rate: {overall_rate:.1f} cand/s | "
                  f"ETA {eta_min:.1f}min", flush=True)
                  
            prev_idx = end_idx
            prev_time = now

    # Flush memmap
    embeddings.flush()
    t_embed_end = time.time()
    total_time = t_embed_end - t_embed_start
    print(f"\n[embed] Embedding complete: {total:,} candidates in {total_time/60:.1f} minutes", flush=True)
    print(f"  Rate: {total / total_time:.1f} candidates/second", flush=True)

    # Verify no NaNs
    print("[embed] Verifying embeddings (NaN check) ...", flush=True)
    emb_array = np.memmap(
        str(OUT_EMBEDDINGS), dtype=np.float32, mode="r",
        shape=(total, EMBEDDING_DIM)
    )
    nan_count = np.isnan(emb_array).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values detected!", flush=True)
    else:
        print(f"  OK: No NaN values. Shape: {emb_array.shape}, dtype: {emb_array.dtype}", flush=True)

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    print("[embed] Building FAISS IndexFlatIP ...", flush=True)
    t_faiss_start = time.time()

    emb_for_faiss = np.array(emb_array)  # copy from memmap
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(emb_for_faiss)

    faiss.write_index(index, str(OUT_FAISS))
    t_faiss_end = time.time()
    print(f"[embed] FAISS index built and saved: ntotal={index.ntotal}, "
          f"time={t_faiss_end - t_faiss_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_wall = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"[embed] COMPLETE", flush=True)
    print(f"  Model: {MODEL_NAME} ({EMBEDDING_DIM}-dim)", flush=True)
    print(f"  Candidates embedded: {total:,}", flush=True)
    print(f"  Total wall-clock time: {total_wall/60:.1f} minutes", flush=True)
    print(f"  Embedding pass time: {total_time/60:.1f} minutes", flush=True)
    print(f"  Artifacts produced:", flush=True)
    for p in [OUT_EMBEDDINGS, OUT_JD_EMBEDDINGS, OUT_FAISS, OUT_IDS]:
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"    {p.name}: {size_mb:.1f} MB", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
