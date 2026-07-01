import os
from pathlib import Path

class ScorerBase:
    """Base class for all candidate scorers in ARIA.
    
    Loads required artifacts once at initialization.
    Scores candidates in the range [0.0, 1.0].
    """
    def __init__(self, artifacts_dir="artifacts/"):
        self.artifacts_dir = Path(artifacts_dir)
        self._load_artifacts()

    def _load_artifacts(self):
        """Hook for loading artifacts once during init."""
        pass

    def score(self, candidate_id: str) -> float:
        """Scores a single candidate. Returns a value in [0.0, 1.0]."""
        raise NotImplementedError("Subclasses must implement score()")

    def score_batch(self, candidate_ids: list[str]) -> dict[str, float]:
        """Scores a batch of candidates. Returns a mapping of candidate_id -> score.
        
        Can be overridden in subclasses for vectorization/batch-level performance.
        """
        return {cid: self.score(cid) for cid in candidate_ids}
