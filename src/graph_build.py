#!/usr/bin/env python3
"""
graph_build.py — Skill co-occurrence graph + JD proximity scoring.

Builds a PMI-weighted NetworkX graph from skill co-occurrence across the
100K candidate pool, then computes per-candidate proximity to JD-relevant
skill nodes using shortest-path distance (inverted + normalized to [0,1]).

Artifacts produced:
  - artifacts/skill_graph.pkl             (NetworkX graph for inspection)
  - artifacts/skill_graph_proximity.npy   (100K floats, same row order as candidate_ids.json)
"""

import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"

OUT_GRAPH = ARTIFACTS_DIR / "skill_graph.pkl"
OUT_PROXIMITY = ARTIFACTS_DIR / "skill_graph_proximity.npy"
CANDIDATE_IDS_FILE = ARTIFACTS_DIR / "candidate_ids.json"
JD_STRUCTURED_FILE = ARTIFACTS_DIR / "jd_structured.json"

# Add src to path
sys.path.insert(0, str(ROOT / "src"))

# PMI edge threshold — only keep edges with PMI > 0 (positive association)
PMI_THRESHOLD = 0.0
# Minimum co-occurrence count to consider an edge (noise filter)
MIN_COOCCURRENCE = 5


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load JD structure
    # ------------------------------------------------------------------
    if not JD_STRUCTURED_FILE.exists():
        print("[graph] jd_structured.json not found — running jd_parser ...")
        from jd_parser import main as jd_main
        jd_main()

    with open(JD_STRUCTURED_FILE, "r", encoding="utf-8") as f:
        jd = json.load(f)

    # ------------------------------------------------------------------
    # Pass 1: Collect skill statistics
    # ------------------------------------------------------------------
    print(f"[graph] Pass 1: Collecting skill co-occurrence from {DATA_FILE} ...")

    skill_count = Counter()         # skill -> number of candidates with this skill
    pair_count = Counter()          # (skill_a, skill_b) -> co-occurrence count (a < b)
    candidate_skills = []           # list of sets, one per candidate
    candidate_ids_from_data = []    # preserve order
    total_candidates = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total_candidates += 1

            skills = cand.get("skills", [])
            skill_names = sorted(set(sk["name"].strip().lower() for sk in skills))
            candidate_skills.append(set(skill_names))
            candidate_ids_from_data.append(cand["candidate_id"])

            for s in skill_names:
                skill_count[s] += 1

            # Count co-occurrences (unordered pairs, sorted to deduplicate)
            for i in range(len(skill_names)):
                for j in range(i + 1, len(skill_names)):
                    pair_count[(skill_names[i], skill_names[j])] += 1

            if total_candidates % 10000 == 0:
                print(f"  ... {total_candidates:,} candidates processed")

    print(f"[graph] Pass 1 complete: {total_candidates:,} candidates, "
          f"{len(skill_count):,} unique skills, {len(pair_count):,} skill pairs")

    # ------------------------------------------------------------------
    # Load canonical candidate_ids ordering (from embed.py)
    # ------------------------------------------------------------------
    if CANDIDATE_IDS_FILE.exists():
        with open(CANDIDATE_IDS_FILE, "r", encoding="utf-8") as f:
            canonical_ids = json.load(f)
        print(f"[graph] Loaded canonical ordering from candidate_ids.json ({len(canonical_ids):,} entries)")
        # Build mapping from candidate_id -> index in our data order
        data_id_to_idx = {cid: i for i, cid in enumerate(candidate_ids_from_data)}
    else:
        print("[graph] WARNING: candidate_ids.json not found — using data file order")
        canonical_ids = candidate_ids_from_data

    # ------------------------------------------------------------------
    # Build PMI-weighted graph
    # ------------------------------------------------------------------
    print("[graph] Computing PMI and building graph ...")

    N = total_candidates
    G = nx.Graph()

    # Add all skills as nodes
    for skill, count in skill_count.items():
        G.add_node(skill, frequency=count, p=count / N)

    # Add edges with PMI weight
    edges_added = 0
    for (a, b), co_count in pair_count.items():
        if co_count < MIN_COOCCURRENCE:
            continue

        p_a = skill_count[a] / N
        p_b = skill_count[b] / N
        p_ab = co_count / N

        # PMI = log2(P(a,b) / (P(a) * P(b)))
        if p_a * p_b > 0:
            pmi = math.log2(p_ab / (p_a * p_b))
            if pmi > PMI_THRESHOLD:
                G.add_edge(a, b, weight=pmi, co_count=co_count)
                edges_added += 1

    print(f"[graph] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save graph
    with open(OUT_GRAPH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[graph] Graph saved to {OUT_GRAPH}")

    # ------------------------------------------------------------------
    # Match JD keywords to graph nodes
    # ------------------------------------------------------------------
    print("[graph] Matching JD keywords to graph nodes ...")

    jd_keywords_raw = set()
    for kw in jd.get("must_have_keywords", []):
        jd_keywords_raw.add(kw.lower().strip())
    for kw in jd.get("nice_to_have_keywords", []):
        jd_keywords_raw.add(kw.lower().strip())

    # Match to graph nodes: exact match first, then substring match
    graph_nodes = set(G.nodes())
    jd_skill_nodes = set()

    for kw in jd_keywords_raw:
        if kw in graph_nodes:
            jd_skill_nodes.add(kw)
        else:
            # Substring match: find graph nodes that contain this keyword
            for node in graph_nodes:
                if kw in node or node in kw:
                    jd_skill_nodes.add(node)

    print(f"[graph] JD skill nodes matched: {len(jd_skill_nodes)} / {len(jd_keywords_raw)} keywords")
    if jd_skill_nodes:
        print(f"  Sample matches: {list(jd_skill_nodes)[:15]}")

    # ------------------------------------------------------------------
    # Compute per-candidate proximity to JD skill nodes
    # ------------------------------------------------------------------
    print("[graph] Computing per-candidate proximity scores ...")

    # Pre-compute shortest path lengths from all JD skill nodes
    # Use BFS-based shortest path (unweighted) for speed
    # For each JD skill node, get shortest path to all reachable nodes
    jd_node_distances = {}  # node -> min distance to any JD skill node
    for jd_node in jd_skill_nodes:
        if jd_node not in G:
            continue
        try:
            lengths = nx.single_source_shortest_path_length(G, jd_node)
            for node, dist in lengths.items():
                if node not in jd_node_distances or dist < jd_node_distances[node]:
                    jd_node_distances[node] = dist
        except nx.NetworkXError:
            continue

    # Maximum distance for normalization
    max_dist = max(jd_node_distances.values()) if jd_node_distances else 1

    # Compute proximity per candidate
    proximity_scores = np.zeros(len(canonical_ids), dtype=np.float32)

    for out_idx, cid in enumerate(canonical_ids):
        # Find this candidate's skills in our data
        if cid in data_id_to_idx if CANDIDATE_IDS_FILE.exists() else True:
            data_idx = data_id_to_idx[cid] if CANDIDATE_IDS_FILE.exists() else out_idx
            cand_skills = candidate_skills[data_idx]
        else:
            cand_skills = set()

        if not cand_skills:
            proximity_scores[out_idx] = 0.0
            continue

        # For each candidate skill, find its min distance to JD skill nodes
        distances = []
        direct_matches = 0
        for skill in cand_skills:
            if skill in jd_skill_nodes:
                direct_matches += 1
                distances.append(0)  # Direct match = distance 0
            elif skill in jd_node_distances:
                distances.append(jd_node_distances[skill])
            # Skills not in graph or not reachable from JD nodes are ignored

        if not distances:
            proximity_scores[out_idx] = 0.0
            continue

        # Score: combine direct matches + proximity
        # Average distance, inverted and normalized
        avg_dist = np.mean(distances)
        # Proximity = 1 - (avg_dist / max_dist), clamped to [0, 1]
        proximity = max(0.0, 1.0 - (avg_dist / max(max_dist, 1)))

        # Boost for direct matches (fraction of JD skills directly matched)
        if len(jd_skill_nodes) > 0:
            direct_match_ratio = direct_matches / len(jd_skill_nodes)
            # Blend: 60% proximity + 40% direct match ratio
            score = 0.6 * proximity + 0.4 * direct_match_ratio
        else:
            score = proximity

        proximity_scores[out_idx] = np.float32(min(1.0, max(0.0, score)))

        if (out_idx + 1) % 20000 == 0:
            print(f"  ... {out_idx + 1:,}/{len(canonical_ids):,} candidates scored")

    # Save
    np.save(OUT_PROXIMITY, proximity_scores)
    t_total = time.time() - t0

    print(f"\n[graph] COMPLETE in {t_total/60:.1f} minutes")
    print(f"  Proximity scores: shape={proximity_scores.shape}, "
          f"dtype={proximity_scores.dtype}")
    print(f"  Range: [{proximity_scores.min():.4f}, {proximity_scores.max():.4f}]")
    print(f"  Mean: {proximity_scores.mean():.4f}, Median: {np.median(proximity_scores):.4f}")
    print(f"  Non-zero: {np.count_nonzero(proximity_scores):,} / {len(proximity_scores):,}")

    # Quick sanity check
    print("\n[graph] Top 10 proximity scores:")
    top_idx = np.argsort(proximity_scores)[-10:][::-1]
    for idx in top_idx:
        cid = canonical_ids[idx]
        data_idx = data_id_to_idx.get(cid, idx)
        skills_list = sorted(candidate_skills[data_idx]) if data_idx < len(candidate_skills) else []
        print(f"  {cid}: {proximity_scores[idx]:.4f} | skills: {skills_list[:8]}...")


if __name__ == "__main__":
    main()
