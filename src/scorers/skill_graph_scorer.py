import json
import numpy as np
from src.scorers.base import ScorerBase

class SkillGraphScorer(ScorerBase):
    """Scores candidates based on their pre-computed skill graph proximity.
    
    This scorer is low-compute; it maps candidate IDs to their pre-computed PMI-based 
    proximity score from the skill graph generated in the first phase.
    """
    def _load_artifacts(self):
        # 1. Load candidate IDs to build row index mapping
        ids_path = self.artifacts_dir / "candidate_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing: {ids_path}")
        with open(ids_path, "r", encoding="utf-8") as f:
            candidate_ids = json.load(f)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}

        # 2. Load pre-computed proximity array
        prox_path = self.artifacts_dir / "skill_graph_proximity.npy"
        if not prox_path.exists():
            raise FileNotFoundError(f"Missing: {prox_path}")
        self.proximity = np.load(prox_path)

    def score(self, candidate_id: str) -> float:
        if candidate_id not in self.id_to_idx:
            return 0.0
        idx = self.id_to_idx[candidate_id]
        # Proximity score is already precomputed in [0.0, 1.0]
        return float(self.proximity[idx])

    def score_batch(self, candidate_ids: list[str]) -> dict[str, float]:
        scores = {}
        for cid in candidate_ids:
            if cid in self.id_to_idx:
                idx = self.id_to_idx[cid]
                scores[cid] = float(self.proximity[idx])
            else:
                scores[cid] = 0.0
        return scores
