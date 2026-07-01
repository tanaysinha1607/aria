import pickle
from src.scorers.base import ScorerBase

class IntegrityScorer(ScorerBase):
    """Scores candidates based on their pre-computed soft integrity checks.
    
    Loads integrity_flags.pkl (a mapping of candidate_id -> integrity metrics, including 'score').
    Returns the integrity score in [0.0, 1.0] directly.
    """
    def _load_artifacts(self):
        # Load precomputed integrity flags dict
        integrity_path = self.artifacts_dir / "integrity_flags.pkl"
        if not integrity_path.exists():
            raise FileNotFoundError(f"Missing: {integrity_path}")
        with open(integrity_path, "rb") as f:
            self.integrity_data = pickle.load(f)

    def score(self, candidate_id: str) -> float:
        if candidate_id not in self.integrity_data:
            # If not in the precomputed set, default to maximum integrity (1.0)
            return 1.0
        
        # Access the integrity record
        record = self.integrity_data[candidate_id]
        
        # Expect either a dictionary with 'score' or the score float directly
        if isinstance(record, dict):
            return float(record.get("score", 1.0))
        elif isinstance(record, (int, float)):
            return float(record)
        return 1.0

    def score_batch(self, candidate_ids: list[str]) -> dict[str, float]:
        scores = {}
        for cid in candidate_ids:
            if cid in self.integrity_data:
                record = self.integrity_data[cid]
                if isinstance(record, dict):
                    scores[cid] = float(record.get("score", 1.0))
                else:
                    scores[cid] = float(record)
            else:
                scores[cid] = 1.0
        return scores
