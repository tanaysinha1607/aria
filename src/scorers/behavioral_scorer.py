import json
import numpy as np
from src.scorers.base import ScorerBase

class BehavioralScorer(ScorerBase):
    """Scores candidates based on platform activity and behavioral clustering metrics.
    
    Loads behavioral_features.npy (shape: (100000, 5)) and behavioral_features_meta.json.
    Applies a weighted average across the 5 clusters:
      - responsiveness (Weight: 0.30): Highly predictive of candidate interest and speed of hire.
      - platform_credibility (Weight: 0.25): Ensures profile legitimacy, active connections, and skill endorsements.
      - availability (Weight: 0.20): Reflects how quickly the candidate can start (notice period) and activity dates.
      - verification (Weight: 0.15): Multi-factor identity validation (verified email, phone, LinkedIn).
      - engagement (Weight: 0.10): Platform views and submission rates (weighted lower due to potential signal noise).
    """
    def _load_artifacts(self):
        # 1. Load candidate IDs to build row index mapping
        ids_path = self.artifacts_dir / "candidate_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing: {ids_path}")
        with open(ids_path, "r", encoding="utf-8") as f:
            candidate_ids = json.load(f)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}

        # 2. Load behavioral features array
        feat_path = self.artifacts_dir / "behavioral_features.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing: {feat_path}")
        self.features = np.load(feat_path)

        # 3. Load meta to find index mapping of columns
        meta_path = self.artifacts_dir / "behavioral_features_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Meta columns: ["availability", "responsiveness", "platform_credibility", "verification", "engagement"]
        cols = meta["columns"]
        self.col_to_idx = {col: idx for idx, col in enumerate(cols)}

        # Validate we have all columns
        required_cols = ["availability", "responsiveness", "platform_credibility", "verification", "engagement"]
        for rc in required_cols:
            if rc not in self.col_to_idx:
                raise ValueError(f"Missing behavioral column in meta: {rc}")

        # Map weights directly by feature column index
        self.weights = np.zeros(len(cols), dtype=np.float32)
        self.weights[self.col_to_idx["responsiveness"]] = 0.30
        self.weights[self.col_to_idx["platform_credibility"]] = 0.25
        self.weights[self.col_to_idx["availability"]] = 0.20
        self.weights[self.col_to_idx["verification"]] = 0.15
        self.weights[self.col_to_idx["engagement"]] = 0.10

    def score(self, candidate_id: str) -> float:
        if candidate_id not in self.id_to_idx:
            return 0.0
        idx = self.id_to_idx[candidate_id]
        
        # Extract candidate behavioral slice (length 5)
        cand_feats = self.features[idx]
        
        # Compute weighted average
        weighted_score = float(np.sum(cand_feats * self.weights))
        return max(0.0, min(1.0, weighted_score))

    def score_batch(self, candidate_ids: list[str]) -> dict[str, float]:
        scores = {}
        valid_indices = []
        valid_cids = []

        for cid in candidate_ids:
            if cid in self.id_to_idx:
                valid_indices.append(self.id_to_idx[cid])
                valid_cids.append(cid)
            else:
                scores[cid] = 0.0

        if not valid_indices:
            return scores

        # Load feature slices for the batch
        sub_feats = self.features[valid_indices]  # Shape: (len(valid_indices), 5)
        
        # Compute dot product along the column axis
        weighted_scores = np.dot(sub_feats, self.weights)

        # Clip values to [0.0, 1.0] and populate dict
        for cid, val in zip(valid_cids, weighted_scores):
            scores[cid] = float(max(0.0, min(1.0, val)))

        return scores
