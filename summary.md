# ARIA — Codebase Summary
> **Adaptive Ranked Intelligence Architecture** | Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge
>
> **Team:** CoffeeNCode (Tanay Sinha + Atharv) | **Submission:** `team_aria.csv` | **Demo:** https://aria-redrob.streamlit.app

---

## High-Level Architecture

ARIA is divided into two halves with a clean **artifact-based contract** as the interface boundary:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  HALF 1 — Data & Artifacts Pipeline (Tanay)                                   │
│  Runs once, pre-computes all heavy assets → writes to artifacts/               │
│                                                                                │
│   explore.py ──► jd_parser.py ──► embed.py ──► graph_build.py                 │
│                                              ──► behavioral_index.py           │
│                                              ──► integrity_score.py            │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │ artifacts/ (binary & JSON files)
┌───────────────────────────────────────▼────────────────────────────────────────┐
│  HALF 2 — Scoring, Fusion & Ranking (Atharv)                                  │
│  Reads only from artifacts/ — CPU-only, ≤5 min for 100K candidates            │
│                                                                                │
│   scorers/  ──► fusion.py ──► reasoning.py ──► rank.py (CLI) / app.py (UI)   │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Fusion Formula
```
final_score = (0.45 × structural + 0.20 × semantic + 0.20 × skill_graph + 0.15 × behavioral)
              × integrity_multiplier
```

---

## Root-Level Files

### `app.py`
**Role:** Central execution file — Streamlit Cloud sandbox demo & interactive UI.

**Function:**
- Serves as the primary entry point for the Streamlit web application deployed at `aria-redrob.streamlit.app`.
- Provides a dark-themed glassmorphism UI (CSS injected via `st.markdown`) with animated gradient title.
- Accepts candidate JSON/JSONL uploads (≤100 profiles) or loads the bundled `sample_candidates.json`.
- Runs the **full ARIA ranking pipeline on-the-fly** (no pre-computed artifacts required in sandbox mode):
  1. Loads `artifacts/jd_structured.json` for JD structure.
  2. Uses `sentence-transformers/all-MiniLM-L6-v2` (cached via `@st.cache_resource`) to embed JD sections and candidate text on-the-fly.
  3. Computes inline behavioral score via `compute_inline_behavioral_score()` — 5 clusters.
  4. Computes inline integrity score via `compute_inline_integrity_score()` — 4 flags.
  5. Computes inline skill score via `compute_inline_skill_score()` — keyword overlap.
  6. Calls `StructuralScorer.score()` and `ReasoningGenerator.generate()` from the `src/` modules.
  7. Fuses all scores, sorts, and renders ranked results as a dataframe with a CSV download button and score distribution bar chart.
- Writes temporary files to satisfy scorer file contracts, and cleans them up in `finally`.

**Key functions:**
- `get_embedding_model()` — cached loader for SentenceTransformer model
- `compute_inline_behavioral_score(cand)` → `float [0,1]` — 5-cluster weighted behavioral aggregation
- `compute_inline_integrity_score(cand)` → `(float, list[str])` — 4-flag penalty system
- `compute_inline_skill_score(cand, must_have_kws, nice_to_have_kws)` → `float [0,1]` — keyword overlap
- `main()` — Streamlit app orchestrator

---

### `rank.py`
**Role:** CLI entrypoint for the production ranking pipeline.

**Function:**
- Exposes a CLI: `python rank.py --candidates ./data/candidates.jsonl --out ./team_aria.csv`
- Loads `FusionEngine` and `ReasoningGenerator` from `src/`.
- Validates candidate IDs against the pre-computed `artifacts/candidate_ids.json` canonical set, skipping any missing.
- Calls `engine.score_all_candidates(valid_cids)` for vectorized batch scoring.
- Sorts descending by score, tie-breaks lexicographically ascending by `candidate_id`.
- Slices the **top 100** ranked candidates.
- Writes the output CSV: `candidate_id, rank, score, reasoning`.
- Runs a **Honeypot Sanity Check** at the end — warns if integrity score < 0.5 candidates exceed 8% of top 100.

**Key functions:**
- `load_candidate_ids(candidates_path)` — streams IDs from both JSONL and JSON formats
- `main()` — full pipeline orchestrator

---

### `requirements.txt`
**Role:** Python dependency specification.

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Array math, dataframes |
| `matplotlib`, `seaborn` | Visualization (explore.py) |
| `scikit-learn` | ML utilities |
| `tqdm` | Progress bars |
| `sentence-transformers` | MiniLM-L6-v2 embedding model |
| `faiss-cpu` | Cosine similarity vector index |
| `torch` | Required by sentence-transformers |
| `networkx` | PMI skill co-occurrence graph |
| `streamlit` | Web UI framework |

---

### `README.md`
**Role:** Project documentation and architecture guide.

**Contents:** Architecture overview, five scoring signals & weights, fusion formula explanation, setup instructions, step-by-step reproduction guide, artifacts contract table (shapes, row order, disk size), project structure tree, and team credits.

---

### `findings.md`
**Role:** Auto-generated EDA output from `src/explore.py`.

**Contents:** Title distribution (top 30), top-50 most common skills, years-of-experience histogram, Redrob signal correlation matrix, proposed behavioral cluster groupings, and suspicious candidate inspection results.

---

### `submission_metadata.yaml`
**Role:** Hackathon submission metadata.

**Contents:** Team name (`CoffeeNCode`), primary contact, GitHub repo URL, sandbox demo link, AI tools declared (`Claude`, `Antigravity IDE`), compute environment, team member roles, and methodology summary.

---

### `team_aria.csv`
**Role:** Final ranked submission file (output of `rank.py`).

**Format:** `candidate_id, rank, score, reasoning` — 100 rows of top-ranked candidates.

---

### `.streamlit/config.toml`
**Role:** Streamlit server and theme configuration.

**Settings:** `maxUploadSize=50` MB, `headless=true`, primary color `#FF4B4B`, background `#0E1117` (dark), text `#FAFAFA`.

---

## `src/` — Pipeline Modules

### `src/explore.py`
**Role:** Exploratory Data Analysis (EDA) — **Step 1** of the data pipeline.

**Function:**
- Streams `data/candidates.jsonl` line-by-line (memory-efficient, no full list in RAM).
- Computes: title distribution, top-50 skills, YoE histogram, single-career-history candidate count.
- Builds a Pearson correlation matrix for 18 numeric `redrob_signals` (using Pandas + scipy).
- Detects suspicious candidates using 4 heuristics: `expert_low_duration`, `high_skill_thin_career`, `high_behavioral_thin_career`, `ai_skills_nontech_title`.
- Writes `findings.md` with all findings, correlation analysis, and behavioral cluster recommendations.
- **Output:** `findings.md`

---

### `src/jd_parser.py`
**Role:** Job Description parser — **Step 2** of the data pipeline.

**Function:**
- Manually structures the fixed `data/job_description.md` into a typed Python dict (no NLP/ML).
- Defines `must_have_signals`, `nice_to_have_signals`, `explicit_disqualifiers`, `ideal_profile_description`, `core_responsibilities`, `location_preferences`, `experience_band`, `must_have_keywords`, `nice_to_have_keywords`, `consulting_firms_list`, `non_tech_disqualifier_titles`.
- Exposes `load_jd()` for import by other modules (falls back to in-memory build if file doesn't exist).
- **Output:** `artifacts/jd_structured.json`

---

### `src/embed.py`
**Role:** Semantic embedding generator + FAISS index builder — **Step 3** of the data pipeline.

**Function:**
- Loads `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-only, ~102 candidates/sec).
- Embeds 3 JD sections: `must_have`, `ideal_profile`, `core_responsibilities` → saves as `jd_section_embeddings.npz`.
- Streams `candidates.jsonl`, builds candidate text as `current_title | skill names` (capped at 150 chars).
- Encodes candidates in batches of 256, writes to a memory-mapped `candidate_embeddings.npy` to minimize RAM pressure.
- Writes `candidate_ids.json` — the **master row ordering** that all other artifacts must match.
- Builds a `faiss.IndexFlatIP` (inner product = cosine similarity for normalized vectors), saves as `faiss_index.bin`.
- Performs NaN validation on embeddings after completion.
- **Outputs:** `candidate_embeddings.npy`, `jd_section_embeddings.npz`, `faiss_index.bin`, `candidate_ids.json`

---

### `src/graph_build.py`
**Role:** Skill co-occurrence graph builder + JD proximity scorer — **Step 4** of the data pipeline.

**Function:**
- **Pass 1:** Streams `candidates.jsonl`, collects skill frequency counts and pairwise co-occurrence counts. Stores per-candidate skill sets in memory.
- **Graph construction:** Builds a `networkx.Graph` with PMI-weighted edges. Includes only pairs with co-occurrence ≥ 5 and PMI > 0.
  - `PMI = log2(P(a,b) / (P(a) × P(b)))`
- **JD matching:** Matches JD must-have and nice-to-have keywords to graph nodes (exact first, then substring).
- **Proximity scoring:** Uses BFS (`nx.single_source_shortest_path_length`) from each JD skill node.
  - Per-candidate score = `0.6 × (1 - avg_dist/max_dist) + 0.4 × direct_match_ratio`, clamped to [0,1].
- Reorders output to match `candidate_ids.json` canonical order.
- **Outputs:** `skill_graph.pkl` (NetworkX graph), `skill_graph_proximity.npy` (100K floats)

---

### `src/behavioral_index.py`
**Role:** Behavioral signal clustering & normalization — **Step 5** of the data pipeline.

**Function:**
- Collapses 23 raw `redrob_signals` into 5 clusters, each normalized to [0,1] using percentile-rank normalization (robust to outliers via `scipy.stats.rankdata`):

| Cluster (col) | Signals | Notes |
|---|---|---|
| **Availability** (0) | `notice_period_days` (↓), `open_to_work_flag`, `days_since_active` (↓) | Lower notice = better; more recent = better |
| **Responsiveness** (1) | `recruiter_response_rate`, `avg_response_time_hours` (↓), `interview_completion_rate` | Lower response time = better |
| **Platform Credibility** (2) | `profile_completeness_score`, `connection_count`, `endorsements_received`, `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d` | `github=-1` treated as 0 |
| **Verification** (3) | `verified_email`, `verified_phone`, `linkedin_connected` | Raw binary average (no percentile) |
| **Engagement** (4) | `profile_views_received_30d`, `applications_submitted_30d`, `offer_acceptance_rate` | `offer_acceptance=-1` treated as 0.5 (neutral) |

- Reorders output to canonical `candidate_ids.json` order.
- **Outputs:** `behavioral_features.npy` (100K × 5, float32), `behavioral_features_meta.json`

---

### `src/integrity_score.py`
**Role:** Honeypot/anomaly detection & integrity pre-computation — **Step 6** of the data pipeline.

**Function:**
- **Pass 1:** Streams all candidates, collects skill-to-career-month ratios and total career months. Computes percentile thresholds from the full pool.
- **Pass 2:** Scores each candidate with 4 flags and cumulative penalties:

| Flag | Detection | Penalty |
|---|---|---|
| `skill_experience_mismatch` | "expert" proficiency with < 6 months duration | 0.15 per skill, max 0.45 |
| `keyword_stuffing` | skill_count/career_months ratio > P95, or >25 skills with ≤2 career entries | 0.25 |
| `behavioral_twin` | behavioral mean > P90 AND total career months < P25 | 0.20 |
| `impossible_timeline` | Overlapping concurrent roles > 6 months | 0.30 per overlap, max 0.60 |

- Final integrity score = `max(0, 1.0 - total_penalty)` (soft multiplier, NOT a hard filter).
- **Output:** `artifacts/integrity_flags.pkl` — `dict[candidate_id → {score: float, flags: list[str]}]`

---

### `src/fusion.py`
**Role:** Fusion engine — combines all 5 scorer signals into a single final score.

**Function:**
- `FusionEngine.__init__()` — instantiates all 5 scorers once at init (lazy artifact loading).
- `score_candidate(candidate_id)` → `float` — single-candidate scoring path.
- `score_all_candidates(candidate_ids)` → `dict[str, float]` — vectorized batch path using each scorer's `score_batch()` API for speed.
- Fusion weights: `structural=0.45`, `skill_graph=0.20`, `semantic=0.20`, `behavioral=0.15`; integrity is multiplicative.
- Formula: `final = (0.45×struct + 0.20×sem + 0.20×graph + 0.15×beh) × integrity`

---

### `src/reasoning.py`
**Role:** Reasoning generator — produces fact-grounded, rank-calibrated candidate justifications.

**Function:**
- `ReasoningGenerator.__init__()` — loads `jd_structured.json`, `integrity_flags.pkl`, and caches all candidates from the data file keyed by `candidate_id`.
- `generate(candidate_id, rank, score, subscores)` → `str` (25–60 words):
  - **Opening:** Rank-tiered template (top/mid/neutral/bottom) using actual YoE, title, and company. Prioritizes pedigree companies (Google, Netflix, Meta, Amazon, Flipkart, Uber, Zomato, Swiggy, LinkedIn).
  - **Middle clause:** Extracts an actual sentence from career history descriptions matching JD must-have keywords. Falls back to listing top JD-relevant skills.
  - **Concern clause:** Conditionally adds location concern, high notice period, out-of-band YoE, integrity flags, or rank-based cutoff notes.
  - **Closing (ranks 1–20 only):** Adds specific behavioral data (notice period, contact verification status).
  - Enforces 25–60 word count bounds; truncates or pads as needed.

---

## `src/scorers/` — Scorer Submodules

### `src/scorers/base.py`
**Role:** Abstract base class for all scorers.

**Interface:**
- `ScorerBase.__init__(artifacts_dir)` — stores path, calls `_load_artifacts()` hook.
- `_load_artifacts()` — empty hook, overridden by subclasses to load their artifact files at init.
- `score(candidate_id) → float` — abstract; must return value in [0.0, 1.0].
- `score_batch(candidate_ids) → dict[str, float]` — default loops over `score()`; subclasses can override for vectorization.

---

### `src/scorers/semantic_scorer.py`
**Role:** Embedding-based semantic similarity scorer. **Weight: 0.20**

**Function:**
- Loads `candidate_ids.json`, memory-maps `candidate_embeddings.npy` (100K × 384, float32), loads 3 JD section vectors from `jd_section_embeddings.npz`.
- `score()` — computes dot product (= cosine similarity for L2-normalized vectors) between candidate embedding and JD section vectors. Weighted sum: `0.50×must_have + 0.30×ideal_profile + 0.20×core_responsibilities`. Clipped to [0,1].
- `score_batch()` — vectorized: slices embedding sub-matrix, `np.dot` against all 3 JD vectors simultaneously.

---

### `src/scorers/skill_graph_scorer.py`
**Role:** Pre-computed PMI graph proximity scorer. **Weight: 0.20**

**Function:**
- Loads `candidate_ids.json` and `skill_graph_proximity.npy` (100K floats, already in [0,1]).
- `score()` — pure O(1) lookup into the pre-computed proximity array.
- `score_batch()` — iterates lookups (no further vectorization needed).

---

### `src/scorers/behavioral_scorer.py`
**Role:** Platform behavioral cluster scorer. **Weight: 0.15**

**Function:**
- Loads `candidate_ids.json`, `behavioral_features.npy` (100K × 5), `behavioral_features_meta.json`.
- Cluster weights applied: `responsiveness=0.30`, `platform_credibility=0.25`, `availability=0.20`, `verification=0.15`, `engagement=0.10`.
- `score()` — looks up the 5-feature row, computes `np.sum(features × weights)`.
- `score_batch()` — vectorized: slices feature sub-matrix, `np.dot(sub_feats, weights)`.

---

### `src/scorers/structural_scorer.py`
**Role:** Career-history heuristic scorer — primary signal. **Weight: 0.45**

**Function:**
- On init, reads `jd_structured.json` for consulting firms list, must-have keywords, and non-tech disqualifier titles. Streams and caches all candidate profiles (light structural fields only) from `candidates.jsonl` or `sample_candidates.json`.
- `score(candidate_id)` — 4-component weighted blend + disqualifier multiplier:

| Sub-check | Weight | Description |
|---|---|---|
| **Disqualifier** | ×multiplier | `0.05` if entire career is consulting-only OR AI experience < 12 months, entirely post-2023 (LangChain wrapper) |
| **Experience Band** | 0.20 | `1.0` for 5–9 YoE; linear decay below/above with different slopes |
| **Location** | 0.15 | `1.0` Pune/Noida; `0.85/0.70` welcome cities; `0.60` willing to relocate; `0.30` other India; `0.10` international |
| **Title Alignment** | 0.25 | `0.05` non-tech titles; `1.0` coding/engineering titles; `0.40` pure management; `0.80` neutral |
| **Production Depth** | 0.40 | JD keyword matches + scale indicators in descriptions (`0.6×kw_contrib + 0.4×scale_contrib`) |

---

### `src/scorers/integrity_scorer.py`
**Role:** Integrity multiplier scorer (soft penalty). **Applied multiplicatively.**

**Function:**
- Loads `integrity_flags.pkl` dict at init.
- `score()` — looks up the precomputed integrity score. Returns `1.0` (no penalty) for unknown candidates.
- `score_batch()` — iterates lookups for all candidates.

---

### `src/scorers/__init__.py`
**Role:** Package init — exports all 5 scorer classes + `ScorerBase` via `__all__`.

---

## `tests/` — Test Suite

### `tests/test_artifacts.py`
**Role:** 37-test artifact validation suite (run after the full data pipeline).

| Test Class | What it validates |
|---|---|
| `TestArtifactsExist` | All 10 expected artifact files exist |
| `TestCandidateIds` | Length=100K, all unique, CAND_XXXXXXX format, byte-for-byte order match with `candidates.jsonl` |
| `TestJdStructured` | Required keys present, must_have_signals non-empty |
| `TestCandidateEmbeddings` | Shape (100K, 384), float32, no NaNs, L2-normalized (norm ≈ 1.0) |
| `TestJdSectionEmbeddings` | Has 3 required section keys, dim=384 |
| `TestFaissIndex` | ntotal=100K, d=384 |
| `TestSkillGraphProximity` | Shape (100K,), range [0,1], no NaNs |
| `TestSkillGraphRelativeOrdering` | `{faiss, lora, fine-tuning}` scores higher than `{photoshop}` |
| `TestBehavioralFeatures` | Shape (100K, 5), float32, range [0,1], no NaNs |
| `TestIntegrityFlags` | Count=100K, all IDs present, scores in [0,1], flags are lists |
| `TestBehavioralMeta` | 5 columns with exact expected names |

---

### `tests/test_scorers.py`
**Role:** 9-test scorer validation suite (uses `data/sample_candidates.json`).

| Test | What it validates |
|---|---|
| `test_scorers_initialization` | All 5 scorers instantiate without error |
| `test_semantic_scorer` | [0,1] range, no NaN/Inf, score variation across candidates |
| `test_skill_graph_scorer` | [0,1] range, no NaN |
| `test_behavioral_scorer` | [0,1] range, no NaN |
| `test_integrity_scorer` | [0,1] range, no NaN |
| `test_structural_scorer` | [0,1] range, no NaN |
| `test_structural_consulting_checks` | Mixed consulting+product = NOT disqualified; pure consulting = `multiplier=0.05` |
| `test_structural_missing_descriptions` | `None` and non-string descriptions don't crash scorer |
| `test_structural_scorer_eyeball_comparison` | Top-scoring candidate scores higher than bottom-scoring candidate |

---

## `data/` — Reference Data

| File | Role |
|---|---|
| `candidates.jsonl` | 100K candidate profiles (487 MB, gitignored) — primary input |
| `sample_candidates.json` | First 50 candidates for testing and sandbox demo |
| `job_description.md` | Full Senior AI Engineer JD text (fixed document) |
| `redrob_signals_doc.md` | Documentation for all `redrob_signals` fields |
| `candidate_schema.json` | JSON Schema for candidate profile validation |
| `submission_spec.md` | Hackathon submission rules and constraints |
| `sample_submission.csv` | Reference format for the submission CSV |
| `validate_submission.py` | Official format validator |
| `submission_metadata_template.yaml` | Template for the submission metadata YAML |
| `README_bundle.md` | Data bundle README |

### `data/validate_submission.py`
**Function:** Validates a submission CSV against hackathon rules:
- Header must be exactly: `candidate_id, rank, score, reasoning`
- Exactly 100 data rows
- `candidate_id` format: `CAND_XXXXXXX` (7 digits), no duplicates
- Ranks 1–100 each appearing exactly once, no duplicates
- Scores must be non-increasing by rank
- Tie-break: equal scores ordered by `candidate_id` ascending

---

## `artifacts/` — Generated Artifacts

All artifacts share a single canonical row ordering defined by `candidate_ids.json`.

| Artifact | Shape / Type | Description |
|---|---|---|
| `candidate_ids.json` | `list[str]`, len=100K | **Master ordering contract** — all array artifacts align to this |
| `jd_structured.json` | `dict` | Parsed JD: must-haves, disqualifiers, keywords, location prefs (tracked in git) |
| `candidate_embeddings.npy` | `(100K, 384) float32` | L2-normalized MiniLM-L6-v2 candidate embeddings |
| `jd_section_embeddings.npz` | 3×`(1, 384) float32` | JD section embeddings (must_have, ideal_profile, core_responsibilities) |
| `faiss_index.bin` | `IndexFlatIP`, ntotal=100K | Cosine similarity FAISS index |
| `skill_graph.pkl` | `networkx.Graph` | PMI-weighted skill co-occurrence graph |
| `skill_graph_proximity.npy` | `(100K,) float32` | Per-candidate JD proximity score [0,1] |
| `behavioral_features.npy` | `(100K, 5) float32` | 5 behavioral cluster scores [0,1] |
| `behavioral_features_meta.json` | `dict` | Column names, normalization details, weights |
| `integrity_flags.pkl` | `dict[str → {score, flags}]` | Integrity score + triggered flag names per candidate |

**Total disk:** ~330–380 MB

---

## Full Data Flow

```
data/candidates.jsonl (100K profiles, 487 MB)
        │
        ├──► src/explore.py ─────────────────────────► findings.md
        │
        ├──► src/jd_parser.py ───────────────────────► artifacts/jd_structured.json
        │
        ├──► src/embed.py (MiniLM-L6-v2, ~16 min) ──► artifacts/candidate_embeddings.npy
        │                                           ──► artifacts/jd_section_embeddings.npz
        │                                           ──► artifacts/faiss_index.bin
        │                                           ──► artifacts/candidate_ids.json (row-order master)
        │
        ├──► src/graph_build.py (PMI graph) ─────────► artifacts/skill_graph.pkl
        │                                           ──► artifacts/skill_graph_proximity.npy
        │
        ├──► src/behavioral_index.py (5 clusters) ──► artifacts/behavioral_features.npy
        │                                           ──► artifacts/behavioral_features_meta.json
        │
        └──► src/integrity_score.py (4 flags) ──────► artifacts/integrity_flags.pkl
                      │
                      │  ← artifact boundary →
                      ▼
        rank.py / app.py
           └──► src/fusion.py (FusionEngine)
                   ├── SemanticScorer     (w=0.20) — dot product on embeddings
                   ├── SkillGraphScorer   (w=0.20) — proximity array lookup
                   ├── BehavioralScorer   (w=0.15) — weighted cluster dot product
                   ├── StructuralScorer   (w=0.45) — heuristic career scoring
                   └── IntegrityScorer    (×mult)  — soft multiplicative penalty
                            │
                            ▼
                   src/reasoning.py (ReasoningGenerator)
                            │
                            ▼
                   team_aria.csv (top-100 ranked candidates)
```

---

## Performance Metrics

| Metric | Value |
|---|---|
| Total candidates scored | 100,000 |
| Ranking runtime (CPU-only) | ~42 seconds |
| Honeypot flags in top 100 | 0 / 100 |
| Official validator result | ✅ Submission is valid |
| Embedding pre-compute time | ~16 min (all-MiniLM-L6-v2, CPU-only) |
| Embedding throughput | ~102 candidates/sec on CPU |
| Artifacts total disk | ~330–380 MB |
| Compute environment | Windows 11, Intel i7, 16 GB RAM, CPU-only, Python 3.11 |
