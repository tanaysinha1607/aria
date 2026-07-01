import json
import numpy as np
from pathlib import Path
from src.scorers.base import ScorerBase

class SemanticScorer(ScorerBase):
    """Scores candidates based on semantic embedding similarity to the Job Description (JD).
    
    Loads pre-normalized MiniLM-L6-v2 candidate embeddings and weights their similarity 
    against three parsed JD sections:
      - must_have (Weight: 0.50) - represents hard requirements.
      - ideal_profile (Weight: 0.30) - represents preferred traits and experience.
      - core_responsibilities (Weight: 0.20) - represents day-to-day role tasks.
      
    Because pre-computed embeddings are normalized to unit length, cosine similarity 
    is computed as a dot product. Output is clipped to [0.0, 1.0].
    """
    def _load_artifacts(self):
        # 1. Load candidate IDs to build row index mapping
        ids_path = self.artifacts_dir / "candidate_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing: {ids_path}")
        with open(ids_path, "r", encoding="utf-8") as f:
            candidate_ids = json.load(f)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}
        self.total_candidates = len(candidate_ids)

        # 2. Memory-map candidate embeddings to be RAM-efficient
        emb_path = self.artifacts_dir / "candidate_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing: {emb_path}")
        
        # Dimensions are hardcoded to 384 for all-MiniLM-L6-v2
        self.emb_dim = 384
        self.embeddings = np.memmap(
            str(emb_path), 
            dtype=np.float32, 
            mode="r", 
            shape=(self.total_candidates, self.emb_dim)
        )

        # 3. Load Job Description section embeddings
        jd_path = self.artifacts_dir / "jd_section_embeddings.npz"
        if not jd_path.exists():
            raise FileNotFoundError(f"Missing: {jd_path}")
        jd_data = np.load(jd_path)
        
        # Section vectors are 1D arrays of size 384, squeeze them to (384,)
        self.jd_must_have = np.squeeze(jd_data["must_have"])
        self.jd_ideal_profile = np.squeeze(jd_data["ideal_profile"])
        self.jd_core_responsibilities = np.squeeze(jd_data["core_responsibilities"])

        # Define weights
        self.w_must_have = 0.50
        self.w_ideal_profile = 0.30
        self.w_core_responsibilities = 0.20

    def score(self, candidate_id: str) -> float:
        if candidate_id not in self.id_to_idx:
            # Candidate ID not recognized; return a baseline score of 0.0
            return 0.0
        
        idx = self.id_to_idx[candidate_id]
        # Retrieve candidate embedding slice
        cand_emb = self.embeddings[idx]

        # Vectors are L2-normalized, so dot product = cosine similarity
        sim_must = float(np.dot(cand_emb, self.jd_must_have))
        sim_ideal = float(np.dot(cand_emb, self.jd_ideal_profile))
        sim_core = float(np.dot(cand_emb, self.jd_core_responsibilities))

        # Weighted combination
        weighted_sim = (
            self.w_must_have * sim_must +
            self.w_ideal_profile * sim_ideal +
            self.w_core_responsibilities * sim_core
        )

        # Cosine similarity typically falls in [-1, 1], but clip to [0.0, 1.0]
        return max(0.0, min(1.0, weighted_sim))

    def score_batch(self, candidate_ids: list[str]) -> dict[str, float]:
        """Vectorized batch computation of semantic similarity scores."""
        # Find valid indexes and candidate ids
        valid_indices = []
        valid_cids = []
        scores = {}

        for cid in candidate_ids:
            if cid in self.id_to_idx:
                valid_indices.append(self.id_to_idx[cid])
                valid_cids.append(cid)
            else:
                scores[cid] = 0.0

        if not valid_indices:
            return scores

        # Load valid embeddings slice into memory
        sub_embs = self.embeddings[valid_indices]  # Shape: (len(valid_indices), 384)

        # Dot product against section vectors (Shape: (384,))
        sim_must = np.dot(sub_embs, self.jd_must_have)
        sim_ideal = np.dot(sub_embs, self.jd_ideal_profile)
        sim_core = np.dot(sub_embs, self.jd_core_responsibilities)

        # Weighted sum
        weighted_sims = (
            self.w_must_have * sim_must +
            self.w_ideal_profile * sim_ideal +
            self.w_core_responsibilities * sim_core
        )

        # Clip values to [0.0, 1.0] and populate dict
        for cid, val in zip(valid_cids, weighted_sims):
            scores[cid] = float(max(0.0, min(1.0, val)))

        return scores
