#!/usr/bin/env python3
"""
jd_parser.py — Parse job_description.md into structured sections.

The JD is a fixed document — this module does careful manual structuring,
not NLP parsing. The structured output drives embedding (section-aware),
graph proximity (skill keyword matching), and downstream scorers.

Output: artifacts/jd_structured.json
Exposes: load_jd() for other modules to import.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "jd_structured.json"


def build_jd_structure() -> dict:
    """Return the structured JD as a Python dict."""
    return {
        "must_have_signals": [
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
            "deployed to real users — handling embedding drift, index refresh, "
            "retrieval-quality regression in production",
            "Production experience with vector databases or hybrid search infrastructure "
            "(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS) — "
            "operational experience, not just tutorials",
            "Strong Python with emphasis on code quality",
            "Hands-on experience designing evaluation frameworks for ranking systems "
            "(NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation)"
        ],
        "nice_to_have_signals": [
            "LLM fine-tuning experience (LoRA, QLoRA, PEFT)",
            "Experience with learning-to-rank models (XGBoost-based or neural)",
            "Prior exposure to HR-tech, recruiting tech, or marketplace products",
            "Background in distributed systems or large-scale inference optimization",
            "Open-source contributions in the AI/ML space"
        ],
        "explicit_disqualifiers": [
            "Pure-research-only career (academic labs, research-only roles) "
            "with no production deployment",
            "AI experience consisting primarily of under-12-month LangChain/OpenAI "
            "wrapper projects without pre-LLM-era production ML experience",
            "Senior engineers who haven't written production code in 18+ months "
            "(architecture/tech-lead drift into non-coding roles)",
            "Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, "
            "Cognizant, Capgemini, HCL, Tech Mahindra, Mindtree, Mphasis, "
            "LTIMindtree) with zero product-company experience",
            "Primary expertise in computer vision, speech, or robotics "
            "without significant NLP/IR exposure",
            "5+ years of entirely closed-source proprietary work with zero "
            "external validation (no papers, talks, open-source contributions)"
        ],
        "negative_signals_soft": [
            "Title-chasing trajectory: Senior→Staff→Principal via company-hopping "
            "every ~1.5 years without staying to build",
            "Framework-tutorial-only GitHub/blog profile with no systems thinking "
            "(e.g., LangChain tutorials, 'How I used [framework] to build [demo]')"
        ],
        "ideal_profile_description": (
            "6-8 years total experience, of which 4-5 are in applied ML/AI roles "
            "at product companies (not pure services). Has shipped at least one "
            "end-to-end ranking, search, or recommendation system to real users "
            "at meaningful scale. Has strong opinions about retrieval (hybrid vs dense), "
            "evaluation (offline vs online), and LLM integration (when to fine-tune "
            "vs prompt) — and can defend them with reference to systems they actually "
            "built. Located in or willing to relocate to Noida or Pune. Active on "
            "Redrob platform or has clear signal of being in the job market."
        ),
        "core_responsibilities": (
            "Own the intelligence layer of Redrob's product: ranking, retrieval, "
            "and matching systems that decide what recruiters see when searching "
            "for candidates and what candidates see when searching for roles. "
            "Audit existing BM25 + rule-based scoring. Ship a v2 ranking system "
            "with embeddings, hybrid retrieval, and LLM-based re-ranking. "
            "Set up evaluation infrastructure — offline benchmarks, online A/B "
            "testing, recruiter-feedback loops. Drive long-term architecture for "
            "candidate-JD matching at scale. Mentor next round of hires."
        ),
        "location_preferences": {
            "preferred": ["Pune", "Noida"],
            "welcome": [
                "Hyderabad", "Mumbai", "Delhi NCR", "Delhi",
                "Gurgaon", "Gurugram", "Bangalore", "Bengaluru",
                "New Delhi", "Greater Noida", "Ghaziabad", "Faridabad"
            ],
            "india_other": True,
            "outside_india": "case-by-case, no visa sponsorship"
        },
        "experience_band": {
            "soft_range_years": [5, 9],
            "note": (
                "Not a hard cutoff — the JD explicitly says it will consider "
                "candidates outside this band if other signals are strong. "
                "Some people hit senior judgment at 4 years; some never hit it at 15."
            )
        },
        "must_have_keywords": [
            "embeddings", "retrieval", "ranking", "vector database", "hybrid search",
            "recommendation system", "search system", "FAISS", "Pinecone", "Weaviate",
            "Qdrant", "Milvus", "OpenSearch", "Elasticsearch",
            "NDCG", "MRR", "MAP", "A/B testing", "evaluation framework",
            "sentence-transformers", "production ML", "Python",
            "information retrieval", "search relevance", "re-ranking",
            "candidate matching", "talent matching"
        ],
        "nice_to_have_keywords": [
            "LoRA", "QLoRA", "PEFT", "fine-tuning", "learning-to-rank",
            "XGBoost", "LambdaMART", "HR-tech", "recruiting tech",
            "distributed systems", "inference optimization",
            "open-source", "OSS contributions"
        ],
        "consulting_firms_list": [
            "TCS", "Tata Consultancy Services", "Infosys", "Wipro",
            "Accenture", "Cognizant", "Capgemini", "HCL",
            "HCL Technologies", "Tech Mahindra", "Mindtree",
            "Mphasis", "L&T Infotech", "LTIMindtree", "LTI",
            "Hexaware", "Zensar", "Persistent Systems",
            "NIIT Technologies", "Cyient"
        ],
        "non_tech_disqualifier_titles": [
            "marketing", "sales", "human resources", "hr manager",
            "recruiter", "talent acquisition", "business development",
            "account manager", "content writer", "graphic designer",
            "operations manager", "finance", "admin", "customer support",
            "relationship manager", "brand manager"
        ]
    }


def load_jd() -> dict:
    """Load the structured JD from artifacts/jd_structured.json.

    Falls back to building it in-memory if the file doesn't exist yet.
    """
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_jd_structure()


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    jd = build_jd_structure()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(jd, f, indent=2, ensure_ascii=False)

    print(f"[jd_parser] Written structured JD to {OUTPUT_FILE}")
    print(f"  must_have_signals: {len(jd['must_have_signals'])} items")
    print(f"  nice_to_have_signals: {len(jd['nice_to_have_signals'])} items")
    print(f"  explicit_disqualifiers: {len(jd['explicit_disqualifiers'])} items")
    print(f"  must_have_keywords: {len(jd['must_have_keywords'])} keywords")
    print(f"  nice_to_have_keywords: {len(jd['nice_to_have_keywords'])} keywords")
    print(f"  consulting_firms_list: {len(jd['consulting_firms_list'])} firms")


if __name__ == "__main__":
    main()
