import json
import re
from pathlib import Path
from datetime import datetime
from src.scorers.base import ScorerBase

class StructuralScorer(ScorerBase):
    """Primary scorer for candidates based on structural resume matching from candidates.jsonl.
    
    To avoid OOM and keep lookups O(1) during ranking, we parse the candidates data source
    once at init and cache only the light fields required for the heuristics.
    
    Sub-checks computed:
      1. Disqualifier Check (Applied as a hard multiplier = 0.05 if triggered):
         - Consulting firm only: career_history is entirely spent at consulting/outsourcing firms.
         - LangChain wrapper developer: AI/ML experience is <12 months entirely in the last 12 months,
           with no pre-LLM-era (pre-2023) production ML experience.
      2. Experience Band Fit (Weight: 0.20): Soft penalty outside target [5, 9] years of experience.
      3. Location Match (Weight: 0.15): Noida/Pune preferred, welcome cities scored next, else relocations.
      4. Title/Seniority Alignment (Weight: 0.25): Coding engineering titles vs management-only/non-tech.
      5. Production & Scale Evidence (Weight: 0.40): Keyword matches + scale indicators in descriptions.
    """
    def __init__(self, artifacts_dir="artifacts/", data_file=None):
        self.data_file = data_file
        super().__init__(artifacts_dir=artifacts_dir)

    def _load_artifacts(self):
        # 1. Read the parsed Job Description signals
        jd_path = self.artifacts_dir / "jd_structured.json"
        if not jd_path.exists():
            raise FileNotFoundError(f"Missing structured JD: {jd_path}")
        with open(jd_path, "r", encoding="utf-8") as f:
            self.jd = json.load(f)

        self.consulting_firms = set(firm.lower().strip() for firm in self.jd.get("consulting_firms_list", []))
        self.must_have_keywords = self.jd.get("must_have_keywords", [])
        self.non_tech_titles = set(title.lower().strip() for title in self.jd.get("non_tech_disqualifier_titles", []))
        
        # Experience target range
        exp_band = self.jd.get("experience_band", {})
        self.min_exp, self.max_exp = exp_band.get("soft_range_years", [5, 9])

        # 2. Determine data file path (fallbacks for tests vs production)
        ROOT = Path(__file__).resolve().parent.parent.parent
        possible_paths = []
        if self.data_file:
            possible_paths.append(Path(self.data_file))
        possible_paths.extend([
            ROOT / "data" / "candidates.jsonl",
            ROOT / "data" / "sample_candidates.json",
            Path("data/candidates.jsonl"),
            Path("data/sample_candidates.json")
        ])

        target_data_file = None
        for p in possible_paths:
            if p.exists():
                target_data_file = p
                break

        if not target_data_file:
            raise FileNotFoundError("Could not find candidates.jsonl or sample_candidates.json in search paths.")

        # 3. Load and cache only the light structural fields per candidate in memory
        self.candidates_cache = {}
        print(f"[StructuralScorer] Caching candidate profiles from {target_data_file} ...", flush=True)
        
        if target_data_file.suffix == ".json":
            # Parsing sample JSON array
            with open(target_data_file, "r", encoding="utf-8") as f:
                candidates_list = json.load(f)
            for cand in candidates_list:
                self._cache_candidate(cand)
        else:
            # Parsing streaming JSONL
            with open(target_data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cand = json.loads(line)
                        self._cache_candidate(cand)
                        
        print(f"[StructuralScorer] Cached {len(self.candidates_cache)} profiles.", flush=True)

    def _cache_candidate(self, cand):
        cid = cand["candidate_id"]
        profile = cand.get("profile", {})
        redrob = cand.get("redrob_signals", {})
        
        self.candidates_cache[cid] = {
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "years_of_experience": profile.get("years_of_experience", 0.0),
            "current_title": profile.get("current_title", ""),
            "willing_to_relocate": redrob.get("willing_to_relocate", False),
            "summary": profile.get("summary", ""),
            "career_history": [
                {
                    "company": role.get("company", ""),
                    "title": role.get("title", ""),
                    "start_date": role.get("start_date", ""),
                    "duration_months": role.get("duration_months", 0),
                    "description": role.get("description") # Handle as raw (could be None)
                }
                for role in cand.get("career_history", [])
            ]
        }

    def _is_consulting_company(self, company_name: str) -> bool:
        c_clean = company_name.lower().strip()
        if not c_clean:
            return False
        # Exact match or substring match for well-known firms
        for firm in self.consulting_firms:
            if firm == c_clean:
                return True
            if len(firm) >= 3 and firm in c_clean:
                return True
        return False

    def _is_ai_ml_role(self, title: str, description: str) -> bool:
        t_lower = str(title or "").lower()
        d_lower = str(description or "").lower()

        # Word boundary for short words to avoid false positive substring matches
        for kw in ["ai", "ml", "nlp", "llm", "gpt"]:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, t_lower) or re.search(pattern, d_lower):
                return True

        # Long words substring match
        long_kws = [
            "machine learning", "artificial intelligence", "embedding", "vector", 
            "search", "retrieval", "ranking", "recommend", "neural", "deep learning",
            "pytorch", "tensorflow", "scikit", "classification", "fine-tune", "fine-tuning",
            "lora", "qlora", "peft", "bert", "langchain", "llamaindex", "prompt"
        ]
        for kw in long_kws:
            if kw in t_lower or kw in d_lower:
                return True
        return False

    def _check_disqualifiers(self, profile) -> float:
        """Returns a multiplier (0.05 if disqualified, 1.0 if not)."""
        career = profile["career_history"]
        if not career:
            return 1.0  # Let other score features decay it naturally

        # Disqualifier 1: Consulting firm only
        # True if career history is non-empty and EVERY company is a consulting firm
        entire_career_consulting = True
        for role in career:
            if not self._is_consulting_company(role["company"]):
                entire_career_consulting = False
                break
        
        if entire_career_consulting:
            return 0.05

        # Disqualifier 2: LangChain Wrapper (AI experience <12 months entirely within last 12 months)
        # We classify the candidate as GenAI-only if they have AI/ML roles but none of them started before Jan 1, 2023
        has_ai_experience = False
        has_pre_llm_ai = False
        recent_ai_duration = 0

        for role in career:
            if self._is_ai_ml_role(role["title"], role["description"]):
                has_ai_experience = True
                
                # Check start date
                start_str = role["start_date"]
                is_pre_llm = False
                if start_str:
                    try:
                        # Expect format YYYY-MM-DD
                        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                        if start_dt < datetime(2023, 1, 1):
                            is_pre_llm = True
                    except ValueError:
                        pass
                
                if is_pre_llm:
                    has_pre_llm_ai = True
                else:
                    # Accumulate months for post-LLM/recent AI work
                    recent_ai_duration += role["duration_months"]

        # Disqualify if they have AI experience but it is entirely post-LLM AND total duration is < 12 months
        if has_ai_experience and not has_pre_llm_ai and recent_ai_duration < 12:
            return 0.05

        return 1.0

    def _score_experience_band(self, yoe: float) -> float:
        if self.min_exp <= yoe <= self.max_exp:
            return 1.0
        elif yoe < self.min_exp:
            # Linear decay below target range
            return max(0.1, 1.0 - 0.25 * (self.min_exp - yoe))
        else:
            # Soft linear decay above target range
            return max(0.1, 1.0 - 0.10 * (yoe - self.max_exp))

    def _score_location(self, loc: str, country: str, willing_to_relocate: bool) -> float:
        loc_clean = loc.lower().strip()
        country_clean = country.lower().strip()

        # Noida/Pune Preferred
        if any(pref in loc_clean for pref in ["pune", "noida"]):
            return 1.0

        # Welcome cities list from structured JD
        welcome_cities = [
            "hyderabad", "mumbai", "delhi ncr", "delhi", "gurgaon", "gurugram", 
            "bangalore", "bengaluru", "new delhi", "greater noida", "ghaziabad", "faridabad"
        ]
        if any(wel in loc_clean for wel in welcome_cities):
            return 0.85 if willing_to_relocate else 0.70

        # Willing to relocate from other parts of India
        if willing_to_relocate:
            return 0.60

        # Standard baseline for non-relocating
        if "india" in country_clean or any(c in loc_clean for c in ["india", "in"]):
            return 0.30
        
        # International location
        return 0.10

    def _score_title_alignment(self, current_title: str, career: list) -> float:
        current_title_lower = current_title.lower()
        
        # Determine recent titles from career history
        recent_title_lower = ""
        if career:
            recent_title_lower = career[0]["title"].lower()

        # Check for non-tech disqualifier titles
        for bad_title in self.non_tech_titles:
            if bad_title in current_title_lower or bad_title in recent_title_lower:
                return 0.05

        # Check coding/engineering terms
        coding_terms = ["engineer", "developer", "programmer", "scientist", "ml", "ai", "nlp", "researcher", "mts", "fellow"]
        is_coding_current = any(term in current_title_lower for term in coding_terms)
        is_coding_recent = any(term in recent_title_lower for term in coding_terms)

        # Check pure management/architecture drift (director, vp, head, manager, architect without coding terms)
        mgmt_terms = ["manager", "director", "head", "vp", "president", "chief", "cto", "architect"]
        is_mgmt_current = any(term in current_title_lower for term in mgmt_terms)
        is_mgmt_recent = any(term in recent_title_lower for term in mgmt_terms)

        if is_coding_current or is_coding_current:
            return 1.0
        
        if (is_mgmt_current and not is_coding_current) or (is_mgmt_recent and not is_coding_recent):
            return 0.40

        # Neutral fallback
        return 0.80

    def _score_production_depth(self, summary: str, career: list) -> float:
        # Gracefully filter out null or non-string descriptions
        desc_texts = []
        for role in career:
            desc = role.get("description")
            if isinstance(desc, str) and desc.strip():
                desc_texts.append(desc.lower())
                
        combined_text = " ".join(desc_texts) + " " + summary.lower()

        # 1. Match JD keywords
        kw_count = sum(1 for kw in self.must_have_keywords if kw.lower().strip() in combined_text)
        kw_contrib = min(1.0, kw_count / 10.0)

        # 2. Check for scale and production indicators
        scale_terms = [
            "production", "deployed", "scaling", "users", "scale", "performance", 
            "latency", "throughput", "optimized", "million", "billion", "gb", "tb", "real-time"
        ]
        scale_count = sum(1 for term in scale_terms if term in combined_text)
        scale_contrib = min(1.0, scale_count / 4.0)

        # 0.6 weighting for JD keyword coverage, 0.4 for scale indicators
        return 0.6 * kw_contrib + 0.4 * scale_contrib

    def score(self, candidate_id: str) -> float:
        if candidate_id not in self.candidates_cache:
            return 0.0
        
        profile = self.candidates_cache[candidate_id]

        # 1. Disqualifier Check
        disq_multiplier = self._check_disqualifiers(profile)

        # 2. Soft Scoring Features
        score_exp = self._score_experience_band(profile["years_of_experience"])
        score_loc = self._score_location(
            profile["location"], 
            profile["country"], 
            profile["willing_to_relocate"]
        )
        score_title = self._score_title_alignment(
            profile["current_title"], 
            profile["career_history"]
        )
        score_prod = self._score_production_depth(
            profile["summary"], 
            profile["career_history"]
        )

        # 3. Weighted Blend
        raw_weighted_score = (
            0.20 * score_exp +
            0.15 * score_loc +
            0.25 * score_title +
            0.40 * score_prod
        )

        # Apply disqualifier multiplier at the end
        final_score = raw_weighted_score * disq_multiplier
        return max(0.0, min(1.0, final_score))
