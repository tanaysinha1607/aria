import os
from pathlib import Path
from src.scorers.semantic_scorer import SemanticScorer
from src.scorers.skill_graph_scorer import SkillGraphScorer
from src.scorers.behavioral_scorer import BehavioralScorer
from src.scorers.structural_scorer import StructuralScorer
from src.scorers.integrity_scorer import IntegrityScorer

class FusionEngine:
    """Combines candidate scoring metrics using a weighted linear combination 
    and applies a soft multiplicative integrity multiplier.
    
    Starting weights:
      - structural (Weight: 0.45): Primary resumes and direct qualification facts.
      - skill_graph (Weight: 0.20): Skills proximity derived from co-occurrence networks.
      - semantic (Weight: 0.20): Text embedding similarity (kept low due to 150-char cap limit).
      - behavioral (Weight: 0.15): Activity, response rates, and verification metrics.
      
    Multipliers:
      - integrity_multiplier (multiplicative): Crushes candidates who fail soft integrity checks.
    """
    def __init__(self, artifacts_dir="artifacts/", data_file=None):
        self.artifacts_dir = Path(artifacts_dir)
        
        # Instantiate individual scorers once
        self.semantic = SemanticScorer(artifacts_dir=self.artifacts_dir)
        self.skill_graph = SkillGraphScorer(artifacts_dir=self.artifacts_dir)
        self.behavioral = BehavioralScorer(artifacts_dir=self.artifacts_dir)
        self.structural = StructuralScorer(artifacts_dir=self.artifacts_dir, data_file=data_file)
        self.integrity = IntegrityScorer(artifacts_dir=self.artifacts_dir)

        # Define weights
        self.w_semantic = 0.20
        self.w_skill_graph = 0.20
        self.w_behavioral = 0.15
        self.w_structural = 0.45

    def score_candidate(self, candidate_id: str) -> float:
        """Scores a single candidate by calling and combining all scorer signals."""
        s_sem = self.semantic.score(candidate_id)
        s_graph = self.skill_graph.score(candidate_id)
        s_beh = self.behavioral.score(candidate_id)
        s_struct = self.structural.score(candidate_id)
        mult_integ = self.integrity.score(candidate_id)

        combined = (
            self.w_semantic * s_sem +
            self.w_skill_graph * s_graph +
            self.w_behavioral * s_beh +
            self.w_structural * s_struct
        )
        return float(combined * mult_integ)

    def score_all_candidates(self, candidate_ids: list[str]) -> dict[str, float]:
        """Vectorized/batch computation of fusion scores for a list of candidate IDs.
        
        Uses fast batch APIs for the scorers that support them to optimize runtime.
        """
        # Batch query individual scorers
        sem_scores = self.semantic.score_batch(candidate_ids)
        graph_scores = self.skill_graph.score_batch(candidate_ids)
        beh_scores = self.behavioral.score_batch(candidate_ids)
        integ_scores = self.integrity.score_batch(candidate_ids)
        
        # StructuralScorer uses cached lookups, score_batch loop runs instantly
        struct_scores = self.structural.score_batch(candidate_ids)

        fusion_scores = {}
        for cid in candidate_ids:
            s_sem = sem_scores.get(cid, 0.0)
            s_graph = graph_scores.get(cid, 0.0)
            s_beh = beh_scores.get(cid, 0.0)
            s_struct = struct_scores.get(cid, 0.0)
            mult_integ = integ_scores.get(cid, 1.0)

            combined = (
                self.w_semantic * s_sem +
                self.w_skill_graph * s_graph +
                self.w_behavioral * s_beh +
                self.w_structural * s_struct
            )
            fusion_scores[cid] = float(combined * mult_integ)

        return fusion_scores
