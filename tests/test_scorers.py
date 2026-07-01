import json
import math
import numpy as np
import pytest
from pathlib import Path

from src.scorers.semantic_scorer import SemanticScorer
from src.scorers.skill_graph_scorer import SkillGraphScorer
from src.scorers.behavioral_scorer import BehavioralScorer
from src.scorers.structural_scorer import StructuralScorer
from src.scorers.integrity_scorer import IntegrityScorer

# Find project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
SAMPLE_DATA_FILE = DATA_DIR / "sample_candidates.json"

@pytest.fixture(scope="module")
def sample_candidates():
    assert SAMPLE_DATA_FILE.exists(), f"Sample candidates file not found at: {SAMPLE_DATA_FILE}"
    with open(SAMPLE_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def candidate_ids(sample_candidates):
    return [cand["candidate_id"] for cand in sample_candidates]


# ======================================================================
# Test 1: Instantiation and simple sanity checks
# ======================================================================
def test_scorers_initialization():
    sem = SemanticScorer(artifacts_dir=str(ARTIFACTS_DIR))
    graph = SkillGraphScorer(artifacts_dir=str(ARTIFACTS_DIR))
    beh = BehavioralScorer(artifacts_dir=str(ARTIFACTS_DIR))
    struct = StructuralScorer(artifacts_dir=str(ARTIFACTS_DIR), data_file=str(SAMPLE_DATA_FILE))
    integ = IntegrityScorer(artifacts_dir=str(ARTIFACTS_DIR))

    assert sem is not None
    assert graph is not None
    assert beh is not None
    assert struct is not None
    assert integ is not None


# ======================================================================
# Test 2: SemanticScorer validations
# ======================================================================
def test_semantic_scorer(candidate_ids):
    scorer = SemanticScorer(artifacts_dir=str(ARTIFACTS_DIR))
    
    scores = scorer.score_batch(candidate_ids)
    assert len(scores) == len(candidate_ids)

    unique_scores = set()
    for cid in candidate_ids:
        score = scores[cid]
        assert not math.isnan(score), f"SemanticScorer score is NaN for {cid}"
        assert not math.isinf(score), f"SemanticScorer score is Inf for {cid}"
        assert 0.0 <= score <= 1.0, f"SemanticScorer score {score} out of bounds for {cid}"
        unique_scores.add(score)

    # Confirm not all scores are identical
    assert len(unique_scores) > 1, "SemanticScorer returned identical scores for all candidates!"


# ======================================================================
# Test 3: SkillGraphScorer validations
# ======================================================================
def test_skill_graph_scorer(candidate_ids):
    scorer = SkillGraphScorer(artifacts_dir=str(ARTIFACTS_DIR))
    
    scores = scorer.score_batch(candidate_ids)
    assert len(scores) == len(candidate_ids)

    for cid in candidate_ids:
        score = scores[cid]
        assert not math.isnan(score), f"SkillGraphScorer score is NaN for {cid}"
        assert 0.0 <= score <= 1.0, f"SkillGraphScorer score {score} out of bounds for {cid}"


# ======================================================================
# Test 4: BehavioralScorer validations
# ======================================================================
def test_behavioral_scorer(candidate_ids):
    scorer = BehavioralScorer(artifacts_dir=str(ARTIFACTS_DIR))
    
    scores = scorer.score_batch(candidate_ids)
    assert len(scores) == len(candidate_ids)

    for cid in candidate_ids:
        score = scores[cid]
        assert not math.isnan(score), f"BehavioralScorer score is NaN for {cid}"
        assert 0.0 <= score <= 1.0, f"BehavioralScorer score {score} out of bounds for {cid}"


# ======================================================================
# Test 5: IntegrityScorer validations
# ======================================================================
def test_integrity_scorer(candidate_ids):
    scorer = IntegrityScorer(artifacts_dir=str(ARTIFACTS_DIR))
    
    scores = scorer.score_batch(candidate_ids)
    assert len(scores) == len(candidate_ids)

    for cid in candidate_ids:
        score = scores[cid]
        assert not math.isnan(score), f"IntegrityScorer score is NaN for {cid}"
        assert 0.0 <= score <= 1.0, f"IntegrityScorer score {score} out of bounds for {cid}"


# ======================================================================
# Test 6: StructuralScorer validations
# ======================================================================
def test_structural_scorer(sample_candidates, candidate_ids):
    scorer = StructuralScorer(artifacts_dir=str(ARTIFACTS_DIR), data_file=str(SAMPLE_DATA_FILE))
    
    scores = scorer.score_batch(candidate_ids)
    assert len(scores) == len(candidate_ids)

    for cid in candidate_ids:
        score = scores[cid]
        assert not math.isnan(score), f"StructuralScorer score is NaN for {cid}"
        assert 0.0 <= score <= 1.0, f"StructuralScorer score {score} out of bounds for {cid}"


# ======================================================================
# Test 7: Consulting disqualifier checks and mixed consulting+product cases
# ======================================================================
def test_structural_consulting_checks(sample_candidates):
    scorer = StructuralScorer(artifacts_dir=str(ARTIFACTS_DIR), data_file=str(SAMPLE_DATA_FILE))

    # Verify Ira Vora (CAND_0000001) - mixed experience (Mindtree is consulting, Dunder Mifflin is paper/product)
    ira_id = "CAND_0000001"
    ira_profile = scorer.candidates_cache.get(ira_id)
    assert ira_profile is not None
    
    # Confirm they have both Mindtree and Dunder Mifflin
    companies = [role["company"] for role in ira_profile["career_history"]]
    assert "Mindtree" in companies
    assert "Dunder Mifflin" in companies
    
    # Check that the consulting multiplier is NOT applied (should return 1.0, meaning not disqualified)
    mult = scorer._check_disqualifiers(ira_profile)
    assert mult == 1.0, f"Ira Vora was incorrectly disqualified with multiplier {mult}!"
    
    # Now find if there is an ENTIRELY consulting candidate in the sample candidates
    consulting_only_candidate = None
    for cand in sample_candidates:
        cid = cand["candidate_id"]
        profile = scorer.candidates_cache[cid]
        career = profile["career_history"]
        if not career:
            continue
            
        all_consulting = True
        for role in career:
            if not scorer._is_consulting_company(role["company"]):
                all_consulting = False
                break
        if all_consulting:
            consulting_only_candidate = cand
            break

    if consulting_only_candidate:
        cid = consulting_only_candidate["candidate_id"]
        profile = scorer.candidates_cache[cid]
        mult = scorer._check_disqualifiers(profile)
        # Verify it got the 0.05 multiplier
        assert mult == 0.05, f"Candidate {cid} (entirely consulting: {[r['company'] for r in profile['career_history']]}) did not trigger consulting disqualifier!"
        print(f"\n[test_structural] Verified consulting-only candidate: {cid} correctly disqualified with multiplier {mult}")
    else:
        print("\n[test_structural] Note: No entirely consulting candidates found in the 50 sample profiles.")


# ======================================================================
# Test 8: Missing/Null description check
# ======================================================================
def test_structural_missing_descriptions():
    scorer = StructuralScorer(artifacts_dir=str(ARTIFACTS_DIR), data_file=str(SAMPLE_DATA_FILE))
    
    # Construct a synthetic candidate with null/missing/non-string description fields
    synthetic_cand = {
        "candidate_id": "CAND_SYNTHETIC",
        "profile": {
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "summary": "Experienced ML Engineer with embeddings and FAISS experience."
        },
        "career_history": [
            {
                "company": "Product Co",
                "title": "Machine Learning Engineer",
                "start_date": "2024-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "Software",
                "company_size": "50-200",
                "description": None  # Null description!
            },
            {
                "company": "Another Co",
                "title": "Data Scientist",
                "start_date": "2022-01-01",
                "end_date": "2024-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "Software",
                "company_size": "50-200",
                "description": 12345  # Non-string description!
            }
        ],
        "redrob_signals": {
            "willing_to_relocate": True
        }
    }
    
    # Inject synthetic profile directly into scorer cache
    scorer._cache_candidate(synthetic_cand)
    
    # Execute score and ensure it runs successfully without throwing exceptions
    try:
        score = scorer.score("CAND_SYNTHETIC")
        assert 0.0 <= score <= 1.0
        print(f"\n[test_structural] Synthetic candidate with null descriptions scored successfully: {score:.4f}")
    except Exception as e:
        pytest.fail(f"Scorer crashed on null/missing description fields with error: {e}")


# ======================================================================
# Eyeball Sanity Checks for StructuralScorer
# ======================================================================
def test_structural_scorer_eyeball_comparison(sample_candidates):
    scorer = StructuralScorer(artifacts_dir=str(ARTIFACTS_DIR), data_file=str(SAMPLE_DATA_FILE))
    
    # Compute scores for all candidates and sort them
    scored_candidates = []
    for cand in sample_candidates:
        cid = cand["candidate_id"]
        score = scorer.score(cid)
        scored_candidates.append((cid, score, cand))

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*80)
    print("STRUCTURAL SCORER EYEBALL COMPARISON REPORT")
    print("="*80)

    # Print top 2 scoring candidates
    print("\n--- TOP FIT CANDIDATES ---")
    for cid, score, cand in scored_candidates[:2]:
        profile = cand["profile"]
        print(f"ID: {cid} | Structural Score: {score:.4f}")
        print(f"  Name: {profile['anonymized_name']}")
        print(f"  Title: {profile['current_title']}")
        print(f"  Location: {profile['location']} (Country: {profile['country']})")
        print(f"  YOE: {profile['years_of_experience']}")
        print(f"  Summary snippet: {profile['summary'][:150]}...")
        print("  Companies:")
        for role in cand["career_history"]:
            print(f"    - {role['company']} | {role['title']} | Duration: {role['duration_months']} mo")
        print("-" * 50)

    # Print bottom 2 scoring candidates
    print("\n--- BOTTOM FIT CANDIDATES (Or Disqualified) ---")
    for cid, score, cand in scored_candidates[-2:]:
        profile = cand["profile"]
        print(f"ID: {cid} | Structural Score: {score:.4f}")
        print(f"  Name: {profile['anonymized_name']}")
        print(f"  Title: {profile['current_title']}")
        print(f"  Location: {profile['location']} (Country: {profile['country']})")
        print(f"  YOE: {profile['years_of_experience']}")
        print(f"  Summary snippet: {profile['summary'][:150]}...")
        print("  Companies:")
        for role in cand["career_history"]:
            print(f"    - {role['company']} | {role['title']} | Duration: {role['duration_months']} mo")
        print("-" * 50)
    
    print("="*80)

    # Assert that the top-fit candidate scores higher than the bottom-fit candidate
    assert scored_candidates[0][1] > scored_candidates[-1][1]
