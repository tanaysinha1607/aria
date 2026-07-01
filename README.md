# ARIA — Adaptive Ranked Intelligence Architecture
**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

> ARIA ranks 100,000 candidates against a Senior AI Engineer JD using semantic embeddings, skill-graph reasoning, behavioral signal clustering, and integrity anomaly detection — going beyond keyword matching to surface genuine fit.

**Live Sandbox:** https://aria-redrob.streamlit.app  
**Team:** CoffeeNCode (Tanay Sinha + Atharv)  
**Submission file:** `team_aria.csv`

---

## Architecture

ARIA is split into two halves with a clean artifact-based interface contract:

| Half | Owner | What it does |
|---|---|---|
| Data & Artifacts Pipeline | Tanay | Pre-computes all embeddings, graphs, behavioral features, and integrity scores. Produces the `artifacts/` directory. |
| Scoring, Fusion & Ranking | Atharv | Consumes `artifacts/` to score, fuse, rank, and produce the final top-100 CSV. Runs in ≤5 min, ≤16 GB, CPU-only, no network. |

**Interface contract:** Atharv's code reads ONLY from `artifacts/` and `data/`. No coupling to `src/` pipeline modules.

---

## How it Works

### Five Scoring Signals

| Scorer | Weight | What it measures |
|---|---|---|
| `StructuralScorer` | 0.45 | Career history depth, production evidence, location fit, experience band, consulting disqualifier, title/seniority alignment |
| `SemanticScorer` | 0.20 | Cosine similarity between candidate embeddings and JD section embeddings (must-have, ideal profile, core responsibilities) |
| `SkillGraphScorer` | 0.20 | PMI-weighted skill co-occurrence graph proximity to JD-relevant skill nodes |
| `BehavioralScorer` | 0.15 | 5-cluster behavioral signal aggregation (availability, responsiveness, credibility, verification, engagement) |
| `IntegrityScorer` | ×multiplier | Soft multiplicative penalty for honeypot-style anomalies (skill-experience mismatch, keyword stuffing, impossible timelines) |

### Fusion Formula
final_score = (0.45×structural + 0.20×semantic + 0.20×skill_graph + 0.15×behavioral) × integrity_multiplier

Structural evidence is weighted highest to compensate for the embedding limitation — candidate text was capped at 150 chars (title + skills only) due to CPU-only constraints, meaning semantic embeddings can't see career history prose. The structural scorer reads career history descriptions directly to fill this gap.

---

## Results

| Metric | Value |
|---|---|
| Total candidates scored | 100,000 |
| Ranking runtime (CPU-only) | 42 seconds |
| Honeypot flags in top 100 | 0 / 100 |
| Official validator | ✅ Submission is valid |
| Embedding pre-compute time | ~16 min (all-MiniLM-L6-v2, CPU-only) |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/tanaysinha1607/aria.git
cd aria

# Create virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Data setup:** Place `candidates.jsonl` (487 MB, 100K lines) in `data/`. Gitignored due to size. All other data files are already tracked.

---

## Reproducing the Submission

### Step 1 — Run the artifacts pipeline (Tanay's half, one-time pre-compute)

```bash
# 1. Exploratory analysis → findings.md (~2-3 min)
python src/explore.py

# 2. JD parsing → artifacts/jd_structured.json (instant)
python src/jd_parser.py

# 3. Embeddings + FAISS index (~16 min on CPU with all-MiniLM-L6-v2)
python src/embed.py

# 4. Skill graph + proximity scores (~3-5 min)
python src/graph_build.py

# 5. Behavioral feature clustering (~2-3 min)
python src/behavioral_index.py

# 6. Integrity scoring (~3-5 min)
python src/integrity_score.py

# 7. Validate all artifacts
python -m pytest tests/test_artifacts.py -v
```

Total pre-compute time: ~30-40 min on CPU.

### Step 2 — Produce the ranked submission CSV (Atharv's half, ≤5 min)

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./team_aria.csv
```

This single command loads all precomputed artifacts, scores all 100K candidates across five signals, fuses scores, breaks ties by `candidate_id` ascending, and outputs the top-100 ranked CSV with reasoning.

### Step 3 — Validate the output

```bash
python data/validate_submission.py team_aria.csv
```

Expected output: `Submission is valid.`

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

Chosen for CPU-only feasibility under time constraints:
- Throughput: ~102 candidates/sec on CPU, completing 100K in ~16 min
- Embedding matrix: 100K × 384 × 4 bytes ≈ 146 MB
- **Known limitation:** candidate text capped at 150 chars (title + skills only). The `StructuralScorer` compensates by reading career history descriptions directly.

---

## Artifacts Contract

All artifacts in `artifacts/` follow a single canonical row ordering defined by `candidate_ids.json`.

| Artifact | Shape / Type | Row Order | Description |
|---|---|---|---|
| `candidate_ids.json` | list[str], len=100,000 | — (defines order) | Master ordering contract |
| `jd_structured.json` | dict | — | Parsed JD: must-haves, disqualifiers, ideal profile, keywords, location prefs |
| `candidate_embeddings.npy` | (100000, 384) float32 | candidate_ids.json | L2-normalized MiniLM embeddings |
| `jd_section_embeddings.npz` | 3 arrays, each (1, 384) float32 | — | Keys: must_have, ideal_profile, core_responsibilities |
| `faiss_index.bin` | FAISS IndexFlatIP, ntotal=100,000 | candidate_ids.json | Cosine similarity index |
| `skill_graph.pkl` | networkx.Graph | — | PMI-weighted skill co-occurrence graph |
| `skill_graph_proximity.npy` | (100000,) float32, [0,1] | candidate_ids.json | Per-candidate proximity to JD skill nodes |
| `behavioral_features.npy` | (100000, 5) float32, [0,1] | candidate_ids.json | 5 behavioral cluster scores |
| `behavioral_features_meta.json` | dict | — | Column names, signal composition, weights |
| `integrity_flags.pkl` | dict[str → {score, flags}] | keyed by candidate_id | Integrity score [0,1] + triggered flag names |

**Disk usage:** ~330-380 MB total (well within the 5 GB budget).

---

## Project Structure
aria/
├── README.md
├── requirements.txt
├── submission_metadata.yaml        # Team + submission metadata
├── .gitignore
├── rank.py                         # CLI entrypoint → produces submission CSV
├── app.py                          # Streamlit Cloud sandbox demo
├── team_aria.csv                   # Final submission file
├── sample_submission.csv           # Format reference output
├── findings.md                     # Generated by explore.py
├── .streamlit/
│   └── config.toml                 # Streamlit theme + server config
├── data/
│   ├── candidates.jsonl            # 100K candidates (gitignored, 487 MB)
│   ├── sample_candidates.json      # First 50 candidates (tracked)
│   ├── job_description.md          # JD we're ranking against
│   ├── redrob_signals_doc.md       # Behavioral signals reference
│   ├── candidate_schema.json       # JSON schema
│   ├── submission_spec.md          # Hackathon submission rules
│   ├── sample_submission.csv       # Format reference
│   ├── validate_submission.py      # Official format validator
│   └── submission_metadata_template.yaml
├── src/
│   ├── explore.py                  # EDA → findings.md
│   ├── jd_parser.py                # JD structuring → artifacts/jd_structured.json
│   ├── embed.py                    # Embeddings + FAISS → 4 artifacts
│   ├── graph_build.py              # Skill graph → 2 artifacts
│   ├── behavioral_index.py         # Behavioral clustering → 2 artifacts
│   ├── integrity_score.py          # Integrity scoring → 1 artifact
│   ├── fusion.py                   # FusionEngine — combines all 5 scorers
│   ├── reasoning.py                # ReasoningGenerator — per-candidate justifications
│   └── scorers/
│       ├── init.py
│       ├── base.py                 # ScorerBase class
│       ├── semantic_scorer.py      # Embedding cosine similarity
│       ├── skill_graph_scorer.py   # PMI graph proximity
│       ├── behavioral_scorer.py    # Behavioral cluster scoring
│       ├── structural_scorer.py    # Career history heuristics
│       └── integrity_scorer.py     # Anomaly soft penalty
├── artifacts/                      # Generated (binary files gitignored)
│   ├── .gitkeep
│   └── jd_structured.json          # Tracked (small, human-readable)
└── tests/
    ├── test_artifacts.py           # Artifact validation (37 tests)
    └── test_scorers.py             # Scorer validation (9 tests)

---

## Sandbox Demo

A live interactive demo is deployed at **https://aria-redrob.streamlit.app**

Upload any JSON/JSONL file of ≤100 candidates (or use the built-in sample), run the full ranking pipeline, and download the ranked CSV output. Uses on-the-fly scoring (no precomputed artifacts required) with the same model and weights as the production pipeline.

---

## Team

| Name | Role |
|---|---|
| Tanay Sinha | Data pipeline, embeddings, skill graph, behavioral indexing, integrity scoring, scoring architecture, reasoning generation, sandbox deployment |
| Atharv | Fusion engine and ranking pipeline |
