# ARIA — Adaptive Ranked Intelligence Architecture
**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

> ARIA ranks 100,000 candidates against a Senior AI Engineer Job Description using semantic embeddings, skill-graph reasoning, behavioral signal clustering, and integrity anomaly detection — going beyond simple keyword matching to surface genuine candidate fit.

**Live Sandbox Demo:** https://aria-redrob.streamlit.app  
**Team Name:** CoffeeNCode  
**Primary Contact:** Tanay Sinha (tanaysinha1607@gmail.com)  
**Submission Files:** `team_aria.csv` (CSV format) and `team_aria.xlsx` (Excel format)

---

## 1. Architecture Overview

ARIA is divided into two decoupled halves separated by a clean **artifact-based interface contract**:

| Half | Owner | Responsibility & Design |
|---|---|---|
| **Data & Artifacts Pipeline** (Half 1) | Tanay | Pre-computes heavy vector embeddings, co-occurrence graphs, behavioral clusters, and integrity anomaly flags. Outputs static asset files to the `artifacts/` folder. |
| **Scoring, Fusion & Ranking** (Half 2) | Atharv | Consumes static pre-computed assets from `artifacts/` to score, fuse, rank, and generate natural reasoning strings for the top 100. Runs offline in $\le$ 42 seconds on CPU. |

> [!NOTE]
> **Decoupled Contract:** Atharv's scoring code reads strictly from the `artifacts/` and `data/` directories, with zero dependencies on Half 1's preprocessing scripts.

---

## 2. How it Works (Scoring Engine)

ARIA ranks candidates using a weighted combination of **five independent signals**:

| Scorer | Combined Weight | Metric & Evaluation Pattern |
|---|---|---|
| `StructuralScorer` | **0.45** | Scans career history text for production metrics, experience band decay, Pune/Noida location preference, coding seniority, and enforces consulting/wrapper disqualifiers. |
| `SemanticScorer` | **0.20** | Measures cosine similarity of candidate titles + skill tags against the JD (weighted as: must-haves 0.50, ideal profile 0.30, core responsibilities 0.20). |
| `SkillGraphScorer` | **0.20** | Measures PMI-weighted proximity between the candidate's skill set and target JD skill nodes on a co-occurrence graph. |
| `BehavioralScorer` | **0.15** | Integrates 23 raw platform signals grouped into 5 clusters (availability, responsiveness, platform credibility, verification, engagement). |
| `IntegrityScorer` | **Multiplier** | Applies a soft multiplicative penalty index derived from honeypot-style anomaly detectors (skill duration mismatch, keyword stuffing, impossible overlapping timelines). |

### Fusion Formula
$$\text{Fused Score} = (0.45 \times \text{structural} + 0.20 \times \text{semantic} + 0.20 \times \text{skill\_graph} + 0.15 \times \text{behavioral}) \times \text{integrity\_multiplier}$$

> [!TIP]
> **Heuristic Rationale:** Since embeddings were generated using `all-MiniLM-L6-v2` with candidate text capped at 150 characters (limiting semantic visibility into career history prose), the `StructuralScorer` is weighted highest (0.45) to ensure specific career evidence determines rank.

---

## 3. Results Summary

*   **Total Candidates Processed:** 100,000
*   **Pipeline Wall-Clock Runtime:** **31.74 seconds** on CPU (well under the 5-minute sandbox limit).
*   **Honeypot Rate in Top 100:** **0%** (0 / 100 candidates triggered integrity flags, avoiding disqualification).
*   **Official Validator Output:** `Submission is valid.`

---

## 4. Setup & Installation

```bash
# Clone the repository
git clone https://github.com/tanaysinha1607/aria.git
cd aria

# Create and activate Python virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell/CMD)
# source .venv/bin/activate   # Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Dataset Setup:** Place the raw candidate file `candidates.jsonl` (487 MB, 100K profiles) inside the `data/` folder. This file is gitignored due to size limits.

---

## 5. Reproducing the Ranking Pipeline

### Step A — Pre-compute Pipeline (Tanay's Half, One-Time Run)
Run these scripts sequentially to regenerate all pre-computed artifacts:
```bash
# 1. Run exploratory data analysis and generate findings.md (~2 min)
python src/explore.py

# 2. Parse job description into structured JSON (instant)
python src/jd_parser.py

# 3. Compute vector embeddings and FAISS index (~16 min on CPU)
python src/embed.py

# 4. Generate PMI-weighted skill graph and candidate proximity scores (~3 min)
python src/graph_build.py

# 5. Extract and normalize behavioral cluster features (~2 min)
python src/behavioral_index.py

# 6. Evaluate anomaly flag penalties and integrity scores (~3 min)
python src/integrity_score.py

# 7. Run test suite to validate all generated binary and JSON assets
python -m pytest tests/test_artifacts.py -v
```

### Step B — Rank Candidates and Output CSV (Atharv's Half)
To rank the candidates and output the final validated submission files:
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./team_aria.csv
```

### Step C — Format Validation
Verify that the output file satisfies all Redrob submission format constraints:
```bash
python data/validate_submission.py team_aria.csv
```
Expected output: `Submission is valid.`

---

## 6. Artifacts Contract Summary

All pre-computed scoring matrices follow a canonical row-ordering contract mapped strictly to `candidate_ids.json`.

| Artifact Name | Shape / Type | Mapping Reference | Description |
|---|---|---|---|
| `candidate_ids.json` | `list[str]`, len=100,000 | Master Order | Defines the exact row index map for all downstream tensors. |
| `jd_structured.json` | `dict` (JSON) | — | Parsed job requirements, preferences, and keywords. |
| `candidate_embeddings.npy` | `(100000, 384)` float32 | `candidate_ids.json` | Normalized sentence embeddings (`all-MiniLM-L6-v2`). |
| `jd_section_embeddings.npz`| 3 vectors of shape `(1, 384)` | — | Segmented JD vector embeddings. |
| `faiss_index.bin` | FAISS `IndexFlatIP` | `candidate_ids.json` | Pre-indexed cosine similarity search structure. |
| `skill_graph.pkl` | NetworkX Graph | — | Co-occurrence graph of skills. |
| `skill_graph_proximity.npy`| `(100000,)` float32 | `candidate_ids.json` | PMI proximity scores to target skills. |
| `behavioral_features.npy`  | `(100000, 5)` float32 | `candidate_ids.json` | Normalized behavioral cluster coordinates. |
| `behavioral_features_meta.json`| `dict` (JSON) | — | Behavioral metrics normalization factors. |
| `integrity_flags.pkl` | `dict[str -> {score, flags}]` | Keyed by Candidate ID | Anomalous flag mappings for soft penaltization. |

---

## 7. Codebase Structure

```
aria/
├── README.md
├── requirements.txt
├── submission_metadata.yaml        # Team and submission metadata
├── .gitignore
├── rank.py                         # CLI entrypoint -> produces submission CSV
├── app.py                          # Streamlit Cloud sandbox demo dashboard
├── team_aria.csv                   # Final submission file (CSV format)
├── team_aria.xlsx                  # Final submission file (Excel format)
├── sample_submission.csv           # Reference format structure
├── findings.md                     # EDA analysis findings output
├── .streamlit/
│   └── config.toml                 # Streamlit client & theme settings
├── data/
│   ├── candidates.jsonl            # 100K candidates (Gitignored, 487 MB)
│   ├── sample_candidates.json      # First 50 candidates batch (Tracked)
│   ├── job_description.md          # Candidate targets JDs
│   ├── redrob_signals_doc.md       # Behavioral clusters documentation
│   ├── candidate_schema.json       # Candidate validation JSON schema
│   ├── submission_spec.md          # Challenge submission rules
│   ├── sample_submission.csv       # Format sample reference
│   ├── validate_submission.py      # Format check utility
│   └── submission_metadata_template.yaml
├── src/
│   ├── explore.py                  # Profiling exploration -> findings.md
│   ├── jd_parser.py                # JD extraction -> artifacts/jd_structured.json
│   ├── embed.py                    # Embeddings & FAISS mapping
│   ├── graph_build.py              # PMI graph engineering
│   ├── behavioral_index.py         # Behavioral signal normalization
│   ├── integrity_score.py          # Honeypot checks & flag mappings
│   ├── fusion.py                   # Scoring fusion logic
│   ├── reasoning.py                # Grounded candidate justification generator
│   └── scorers/
│       ├── __init__.py
│       ├── base.py                 # ScorerBase interface class
│       ├── semantic_scorer.py      # Title & skill semantic similarity
│       ├── skill_graph_scorer.py   # Skill-graph shortest path scoring
│       ├── behavioral_scorer.py    # 5-cluster platform signal evaluation
│       ├── structural_scorer.py    # Resume facts, locations, and YOE parser
│       └── integrity_scorer.py     # Multiplicative anomaly scorer
├── artifacts/                      # Static precomputed folder
│   ├── .gitkeep
│   └── jd_structured.json          # Tracked Job Description template JSON
└── tests/
    ├── test_artifacts.py           # Artifact parity validations (37 tests)
    └── test_scorers.py             # Score bounds and edge-cases (9 tests)
```

---

## 8. Sandbox Demo Dashboard

A live interactive sandbox demonstration is deployed and accessible at:  
👉 **[https://aria-redrob.streamlit.app](https://aria-redrob.streamlit.app)**

*   **Offline Simulation:** Streamlit Cloud evaluates candidates dynamically on-the-fly without needing local binary files.
*   **Evaluation Batch:** Supports custom file uploads up to 100 profiles (or use the folder icon to load the pre-configured sample dataset instantly).
*   **Transparency:** Displays rank monotonicity charts, preview tables, status stages, and provides a formatted CSV file download.
