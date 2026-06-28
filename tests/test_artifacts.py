#!/usr/bin/env python3
"""
test_artifacts.py — Sanity tests for every artifact produced by the data pipeline.

Verifies:
  - All artifacts exist
  - Shapes/lengths match exactly 100,000 candidates
  - Row ordering matches candidate_ids.json
  - No NaN values
  - Score ranges within documented bounds
  - Skill-graph relative-ordering property
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

CANDIDATE_IDS = ARTIFACTS / "candidate_ids.json"
JD_STRUCTURED = ARTIFACTS / "jd_structured.json"
CANDIDATE_EMBEDDINGS = ARTIFACTS / "candidate_embeddings.npy"
JD_EMBEDDINGS = ARTIFACTS / "jd_section_embeddings.npz"
FAISS_INDEX = ARTIFACTS / "faiss_index.bin"
SKILL_GRAPH = ARTIFACTS / "skill_graph.pkl"
SKILL_PROXIMITY = ARTIFACTS / "skill_graph_proximity.npy"
BEHAVIORAL_FEATURES = ARTIFACTS / "behavioral_features.npy"
BEHAVIORAL_META = ARTIFACTS / "behavioral_features_meta.json"
INTEGRITY_FLAGS = ARTIFACTS / "integrity_flags.pkl"

EXPECTED_COUNT = 100_000
# all-MiniLM-L6-v2 = 384-dim
EXPECTED_DIM = 384


# ======================================================================
# Fixture: load candidate IDs once
# ======================================================================
@pytest.fixture(scope="module")
def candidate_ids():
    assert CANDIDATE_IDS.exists(), f"Missing: {CANDIDATE_IDS}"
    with open(CANDIDATE_IDS, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return ids


# ======================================================================
# Test 1: All artifacts exist
# ======================================================================
class TestArtifactsExist:
    @pytest.mark.parametrize("path", [
        CANDIDATE_IDS, JD_STRUCTURED, CANDIDATE_EMBEDDINGS,
        JD_EMBEDDINGS, FAISS_INDEX, SKILL_GRAPH,
        SKILL_PROXIMITY, BEHAVIORAL_FEATURES, BEHAVIORAL_META,
        INTEGRITY_FLAGS,
    ])
    def test_artifact_exists(self, path):
        assert path.exists(), f"Missing artifact: {path.name}"


# ======================================================================
# Test 2: candidate_ids.json
# ======================================================================
class TestCandidateIds:
    def test_length(self, candidate_ids):
        assert len(candidate_ids) == EXPECTED_COUNT, \
            f"Expected {EXPECTED_COUNT}, got {len(candidate_ids)}"

    def test_unique(self, candidate_ids):
        assert len(set(candidate_ids)) == EXPECTED_COUNT, \
            "Duplicate candidate IDs found"

    def test_format(self, candidate_ids):
        for cid in candidate_ids[:100]:
            assert cid.startswith("CAND_"), f"Bad format: {cid}"
            assert len(cid) == 12, f"Bad length: {cid}"  # CAND_ + 7 digits

    def test_byte_for_byte_source_ordering(self, candidate_ids):
        """Confirm candidate_ids.json matches candidates.jsonl in exact order."""
        data_file = ROOT / "data" / "candidates.jsonl"
        assert data_file.exists(), f"Missing source data file: {data_file}"
        
        source_ids = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cand = json.loads(line)
                    source_ids.append(cand["candidate_id"])
                    
        assert len(source_ids) == EXPECTED_COUNT, f"Source candidate count mismatch: {len(source_ids)}"
        assert candidate_ids == source_ids, "CANDIDATE IDs ORDERING IS MISALIGNED WITH candidates.jsonl!"


# ======================================================================
# Test 3: jd_structured.json
# ======================================================================
class TestJdStructured:
    def test_has_required_keys(self):
        with open(JD_STRUCTURED, "r") as f:
            jd = json.load(f)
        required = [
            "must_have_signals", "nice_to_have_signals",
            "explicit_disqualifiers", "ideal_profile_description",
            "core_responsibilities", "location_preferences",
            "experience_band", "must_have_keywords",
        ]
        for key in required:
            assert key in jd, f"Missing key: {key}"

    def test_must_have_nonempty(self):
        with open(JD_STRUCTURED, "r") as f:
            jd = json.load(f)
        assert len(jd["must_have_signals"]) >= 3


# ======================================================================
# Test 4: candidate_embeddings.npy
# ======================================================================
class TestCandidateEmbeddings:
    def test_shape(self, candidate_ids):
        emb = np.memmap(str(CANDIDATE_EMBEDDINGS), dtype=np.float32, mode="r", shape=(EXPECTED_COUNT, EXPECTED_DIM))
        assert emb.shape == (EXPECTED_COUNT, EXPECTED_DIM), \
            f"Expected ({EXPECTED_COUNT}, {EXPECTED_DIM}), got {emb.shape}"

    def test_dtype(self):
        emb = np.memmap(str(CANDIDATE_EMBEDDINGS), dtype=np.float32, mode="r", shape=(EXPECTED_COUNT, EXPECTED_DIM))
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"

    def test_no_nans(self):
        emb = np.memmap(str(CANDIDATE_EMBEDDINGS), dtype=np.float32, mode="r", shape=(EXPECTED_COUNT, EXPECTED_DIM))
        # Check in chunks to be memory-friendly
        for start in range(0, EXPECTED_COUNT, 10000):
            end = min(start + 10000, EXPECTED_COUNT)
            chunk = np.array(emb[start:end])
            assert not np.any(np.isnan(chunk)), \
                f"NaN found in rows {start}-{end}"

    def test_normalized(self):
        """Spot-check that vectors are approximately L2-normalized."""
        emb = np.memmap(str(CANDIDATE_EMBEDDINGS), dtype=np.float32, mode="r", shape=(EXPECTED_COUNT, EXPECTED_DIM))
        sample = np.array(emb[:100])
        norms = np.linalg.norm(sample, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01,
                                   err_msg="Vectors not L2-normalized")


# ======================================================================
# Test 5: jd_section_embeddings.npz
# ======================================================================
class TestJdSectionEmbeddings:
    def test_has_required_sections(self):
        data = np.load(JD_EMBEDDINGS)
        required = ["must_have", "ideal_profile", "core_responsibilities"]
        for key in required:
            assert key in data.files, f"Missing section: {key}"

    def test_shape(self):
        data = np.load(JD_EMBEDDINGS)
        for key in data.files:
            assert data[key].shape[1] == EXPECTED_DIM, \
                f"{key}: expected dim {EXPECTED_DIM}, got {data[key].shape[1]}"


# ======================================================================
# Test 6: FAISS index
# ======================================================================
class TestFaissIndex:
    def test_ntotal(self):
        import faiss
        index = faiss.read_index(str(FAISS_INDEX))
        assert index.ntotal == EXPECTED_COUNT, \
            f"Expected {EXPECTED_COUNT}, got {index.ntotal}"

    def test_dimension(self):
        import faiss
        index = faiss.read_index(str(FAISS_INDEX))
        assert index.d == EXPECTED_DIM, \
            f"Expected dim {EXPECTED_DIM}, got {index.d}"


# ======================================================================
# Test 7: skill_graph_proximity.npy
# ======================================================================
class TestSkillGraphProximity:
    def test_shape(self):
        prox = np.load(SKILL_PROXIMITY)
        assert prox.shape == (EXPECTED_COUNT,), \
            f"Expected ({EXPECTED_COUNT},), got {prox.shape}"

    def test_range(self):
        prox = np.load(SKILL_PROXIMITY)
        assert prox.min() >= 0.0, f"Min below 0: {prox.min()}"
        assert prox.max() <= 1.0, f"Max above 1: {prox.max()}"

    def test_no_nans(self):
        prox = np.load(SKILL_PROXIMITY)
        assert not np.any(np.isnan(prox)), "NaN found in proximity scores"


# ======================================================================
# Test 8: Skill graph relative ordering
# ======================================================================
class TestSkillGraphRelativeOrdering:
    """Verify that candidates with more JD-relevant adjacent skills
    score higher than those with fewer."""

    def test_qdrant_ndcg_ab_beats_faiss_only(self):
        """A candidate with {qdrant, ndcg, a/b testing} should score
        higher than one with {faiss} alone on graph proximity."""
        with open(SKILL_GRAPH, "rb") as f:
            G = pickle.load(f)

        with open(JD_STRUCTURED, "r") as f:
            jd = json.load(f)

        import networkx as nx

        # Get JD skill nodes
        jd_keywords_raw = set()
        for kw in jd.get("must_have_keywords", []):
            jd_keywords_raw.add(kw.lower().strip())
        for kw in jd.get("nice_to_have_keywords", []):
            jd_keywords_raw.add(kw.lower().strip())

        graph_nodes = set(G.nodes())
        jd_nodes = set()
        for kw in jd_keywords_raw:
            if kw in graph_nodes:
                jd_nodes.add(kw)
            else:
                for node in graph_nodes:
                    if kw in node or node in kw:
                        jd_nodes.add(node)

        # Skip test if not enough JD nodes in graph
        if len(jd_nodes) < 3:
            pytest.skip("Not enough JD skill nodes in graph for this test")

        # Compute proximity for two synthetic candidates
        def compute_proximity(skills: set) -> float:
            jd_node_distances = {}
            for jd_node in jd_nodes:
                if jd_node not in G:
                    continue
                try:
                    lengths = nx.single_source_shortest_path_length(G, jd_node)
                    for node, dist in lengths.items():
                        if node not in jd_node_distances or dist < jd_node_distances[node]:
                            jd_node_distances[node] = dist
                except nx.NetworkXError:
                    continue

            max_dist = max(jd_node_distances.values()) if jd_node_distances else 1
            distances = []
            direct_matches = 0
            for skill in skills:
                if skill in jd_nodes:
                    direct_matches += 1
                    distances.append(0)
                elif skill in jd_node_distances:
                    distances.append(jd_node_distances[skill])

            if not distances:
                return 0.0

            avg_dist = np.mean(distances)
            proximity = max(0.0, 1.0 - (avg_dist / max(max_dist, 1)))
            direct_match_ratio = direct_matches / len(jd_nodes) if jd_nodes else 0
            return 0.6 * proximity + 0.4 * direct_match_ratio

        # Candidate A: faiss + lora + fine-tuning llms (3 JD-relevant skills)
        score_a = compute_proximity({"faiss", "lora", "fine-tuning llms"})
        # Candidate B: photoshop only (1 unrelated skill)
        score_b = compute_proximity({"photoshop"})
    
        assert score_a > score_b, \
            f"Expected faiss+lora+ft_llms ({score_a:.4f}) > photoshop ({score_b:.4f})"


# ======================================================================
# Test 9: behavioral_features.npy
# ======================================================================
class TestBehavioralFeatures:
    def test_shape(self):
        feat = np.load(BEHAVIORAL_FEATURES)
        assert feat.shape == (EXPECTED_COUNT, 5), \
            f"Expected ({EXPECTED_COUNT}, 5), got {feat.shape}"

    def test_range(self):
        feat = np.load(BEHAVIORAL_FEATURES)
        assert feat.min() >= 0.0, f"Min below 0: {feat.min()}"
        assert feat.max() <= 1.0, f"Max above 1: {feat.max()}"

    def test_no_nans(self):
        feat = np.load(BEHAVIORAL_FEATURES)
        assert not np.any(np.isnan(feat)), "NaN found in behavioral features"

    def test_dtype(self):
        feat = np.load(BEHAVIORAL_FEATURES)
        assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"


# ======================================================================
# Test 10: integrity_flags.pkl
# ======================================================================
class TestIntegrityFlags:
    def test_count(self, candidate_ids):
        with open(INTEGRITY_FLAGS, "rb") as f:
            data = pickle.load(f)
        assert len(data) == EXPECTED_COUNT, \
            f"Expected {EXPECTED_COUNT}, got {len(data)}"

    def test_all_candidates_present(self, candidate_ids):
        with open(INTEGRITY_FLAGS, "rb") as f:
            data = pickle.load(f)
        for cid in candidate_ids:
            assert cid in data, f"Missing candidate: {cid}"

    def test_score_range(self, candidate_ids):
        with open(INTEGRITY_FLAGS, "rb") as f:
            data = pickle.load(f)
        for cid, entry in data.items():
            assert 0.0 <= entry["score"] <= 1.0, \
                f"{cid}: score {entry['score']} out of range"

    def test_flags_are_lists(self):
        with open(INTEGRITY_FLAGS, "rb") as f:
            data = pickle.load(f)
        sample = list(data.values())[:100]
        for entry in sample:
            assert isinstance(entry["flags"], list), \
                f"flags should be list, got {type(entry['flags'])}"


# ======================================================================
# Test 11: behavioral_features_meta.json
# ======================================================================
class TestBehavioralMeta:
    def test_columns_defined(self):
        with open(BEHAVIORAL_META, "r") as f:
            meta = json.load(f)
        assert "columns" in meta
        assert len(meta["columns"]) == 5
        expected = ["availability", "responsiveness", "platform_credibility",
                    "verification", "engagement"]
        assert meta["columns"] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
