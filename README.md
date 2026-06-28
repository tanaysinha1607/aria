# ARIA — Adaptive Ranked Intelligence Architecture

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

ARIA ranks 100,000 candidates against a Senior AI Engineer JD using semantic embeddings, skill-graph reasoning, behavioral signal clustering, and integrity anomaly detection.

## Architecture

This repo is split into two halves:

| Half | Owner | What it does |
|------|-------|-------------|
| **Data & Artifacts Pipeline** (this half) | Tanay | Pre-computes all embeddings, graphs, features, and integrity scores. Produces the `artifacts/` directory. |
| **Scoring, Fusion & Ranking** | Atharv | Consumes `artifacts/` to score, rank, and produce the final top-100 CSV. Runs in ≤5 min, ≤16 GB, CPU-only, no network. |

**Interface contract**: Atharv's code reads ONLY from `artifacts/` and `data/`. No coupling to `src/` modules.

## Setup

```bash
# Clone and enter the repo
cd redrob-aria

# Create virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data setup

Place `candidates.jsonl` (487 MB, 100K lines) in `data/`. The file is gitignored due to size. All other data files (`sample_candidates.json`, `job_description.md`, etc.) are already tracked.

## Reproducing the Artifacts Pipeline

Run these scripts in order from the repo root:

```bash
# 1. Exploratory analysis → findings.md (~2-3 min)
python src/explore.py

# 2. JD parsing → artifacts/jd_structured.json (instant, <1s)
python src/jd_parser.py

# 3. Embeddings + FAISS index → 4 artifacts (~45-90 min on CPU)
python src/embed.py
# For background execution on Windows:
#   start /B python src/embed.py > embed_log.txt 2>&1

# 4. Skill graph + proximity → 2 artifacts (~3-5 min)
python src/graph_build.py

# 5. Behavioral features → 2 artifacts (~2-3 min)
python src/behavioral_index.py

# 6. Integrity scores → 1 artifact (~2-3 min)
python src/integrity_score.py

# 7. Validate all artifacts
python -m pytest tests/test_artifacts.py -v
```

**Total pre-compute time**: ~55-100 min on CPU (dominated by step 3).

### Embedding Model Choice

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

Chosen for fast, CPU-only execution under tight timeline constraints:
- **Throughput**: Runs at **~198.7 candidates/second** on CPU (compared to ~4.4 cand/sec for `bge-base`), completing the entire 100,000 candidate pass in **~8.4 minutes**.
- **Embedding matrix size**: 100K × 384 × 4 bytes ≈ 146 MB. Extremely memory-efficient, preventing any OS page swapping or memory thrashing.
- **Vibe**: Standard symmetric sentence-transformer embedding; no query instruction prefix required.

| Artifact | Shape / Type | Row Order | Description |
|----------|-------------|-----------|-------------|
| `candidate_ids.json` | `list[str]`, len=100,000 | — (defines the order) | Master ordering contract. Every `.npy` array and FAISS index follows this order. |
| `jd_structured.json` | `dict` | — | Parsed JD: must-have/nice-to-have signals, disqualifiers, ideal profile, keywords, location prefs, experience band. |
| `candidate_embeddings.npy` | `(100000, 384)` float32 | `candidate_ids.json` | L2-normalized MiniLM embeddings of candidate text (title + skills capped at 150 chars). |
| `jd_section_embeddings.npz` | 3 named arrays, each `(1, 384)` float32 | — | Keys: `must_have`, `ideal_profile`, `core_responsibilities`. L2-normalized. |
| `faiss_index.bin` | FAISS `IndexFlatIP`, ntotal=100,000 | `candidate_ids.json` | Cosine similarity index (inner product over L2-normalized vectors). |
| `skill_graph.pkl` | `networkx.Graph` | — | PMI-weighted skill co-occurrence graph. Nodes = skill names (lowercased), edges = positive PMI. |
| `skill_graph_proximity.npy` | `(100000,)` float32 | `candidate_ids.json` | Per-candidate proximity to JD skill nodes. Range [0, 1]. Higher = closer. |
| `behavioral_features.npy` | `(100000, 5)` float32 | `candidate_ids.json` | Columns: availability, responsiveness, platform_credibility, verification, engagement. Range [0, 1]. |
| `behavioral_features_meta.json` | `dict` | — | Column names, signal composition, weights, normalization details. |
| `integrity_flags.pkl` | `dict[str → {"score": float, "flags": list[str]}]` | keyed by `candidate_id` | Integrity score [0, 1] (1.0 = clean) + list of triggered flag names. Soft penalty, NOT a hard filter. |

### Disk Budget

| Artifact | Approximate Size |
|----------|-----------------|
| `candidate_embeddings.npy` | ~146 MB |
| `faiss_index.bin` | ~146 MB |
| `skill_graph_proximity.npy` | ~0.4 MB |
| `behavioral_features.npy` | ~1.9 MB |
| `skill_graph.pkl` | ~5-20 MB |
| `integrity_flags.pkl` | ~30-60 MB |
| Others (JSON) | <1 MB |
| **Total** | **~330-380 MB** |

Well within the 5 GB disk budget.

## Project Structure

```
ARIA/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore
├── data/
│   ├── candidates.jsonl            # 100K candidates (gitignored)
│   ├── sample_candidates.json      # First 50 candidates (tracked)
│   ├── job_description.md          # The JD we're ranking against
│   ├── redrob_signals_doc.md       # Behavioral signals reference
│   ├── candidate_schema.json       # JSON schema
│   ├── submission_spec.md          # Hackathon submission rules
│   ├── sample_submission.csv       # Format reference
│   ├── submission_metadata_template.yaml
│   ├── validate_submission.py      # Format validator
│   └── README_bundle.md
├── src/
│   ├── explore.py                  # EDA → findings.md
│   ├── jd_parser.py                # JD structuring → artifacts/jd_structured.json
│   ├── embed.py                    # Embeddings + FAISS → 4 artifacts
│   ├── graph_build.py              # Skill graph → 2 artifacts
│   ├── behavioral_index.py         # Behavioral clustering → 2 artifacts
│   ├── integrity_score.py          # Integrity scoring → 1 artifact
│   └── scorers/                    # Placeholder for Atharv's half
│       └── __init__.py
├── artifacts/                      # Generated outputs (binary files gitignored)
│   └── .gitkeep
├── findings.md                     # Generated by explore.py
└── tests/
    └── test_artifacts.py           # Artifact validation suite
```
