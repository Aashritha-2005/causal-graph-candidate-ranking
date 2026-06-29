"""
Sentence-transformer embeddings for candidates and JD.

Precompute: run once via scripts/precompute.py, artifacts saved to disk.
Ranking step: loads artifacts from disk, zero model inference at ranking time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 256

# JD text assembled from the actual job_description.docx content.
# Structured to emphasise what the JD says it values: production deployment,
# embeddings/retrieval/ranking, product-company experience, eval frameworks.
JD_TEXT = """
Senior AI Engineer — Founding Team. Redrob AI, Series A.
Location: Pune/Noida India hybrid. 5-9 years experience.

What you'd actually be doing: own the intelligence layer — ranking, retrieval,
and matching systems. Ship v2 ranking system with embeddings, hybrid retrieval,
LLM-based re-ranking. Set up evaluation infrastructure — offline benchmarks,
online A/B testing, recruiter-feedback loops.

Things you absolutely need: production experience with embeddings-based retrieval
systems (sentence-transformers, OpenAI embeddings, BGE, E5). Production experience
with vector databases or hybrid search — Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Strong Python. Hands-on experience designing
evaluation frameworks for ranking systems — NDCG, MRR, MAP, A/B test interpretation.

Things we'd like: LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank models.
Prior exposure to HR-tech, recruiting tech, marketplace products.
Background in distributed systems or large-scale inference optimization.
Open-source contributions in AI/ML.

Ideal candidate: 6-8 years total, 4-5 years applied ML/AI at product companies
(not pure services). Shipped end-to-end ranking, search, or recommendation system
to real users at scale. Strong opinions about retrieval (hybrid vs dense),
evaluation (offline vs online), LLM integration (fine-tune vs prompt).
Located in or willing to relocate to Noida, Pune, Hyderabad, Mumbai, Delhi NCR.

What we do NOT want: pure research background, no production deployment.
AI experience limited to LangChain calling OpenAI with no pre-LLM-era ML.
Senior engineers with 18+ months no production code. Title-chasers.
Framework enthusiasts whose GitHub is LangChain tutorials. Entire career at
TCS Infosys Wipro Accenture Cognizant Capgemini with no product company.
CV speech robotics without NLP IR exposure. 5+ years closed-source only.
"""

# JD skill phrases — embedded separately for OT matching (Days 3-4)
JD_SKILLS = [
    "embeddings based retrieval production",
    "vector database FAISS Pinecone Weaviate Qdrant",
    "ranking system NDCG evaluation",
    "sentence transformers semantic search",
    "LLM fine-tuning LoRA QLoRA PEFT",
    "hybrid search dense retrieval BM25",
    "recommendation system shipped production users",
    "A/B testing online offline evaluation",
    "Python machine learning production deployment",
    "NLP natural language processing transformer BERT",
    "learning to rank XGBoost neural ranking",
    "distributed systems large scale inference",
]


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_candidates(
    feats: pd.DataFrame,
    model: SentenceTransformer,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    texts = feats["embed_text"].fillna("").tolist()
    print(f"  Embedding {len(texts):,} candidates (batch={batch_size})...")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-norm for cosine via inner product
    )
    np.save(out_path, vecs.astype(np.float32))
    print(f"  Saved: {out_path}  shape={vecs.shape}")
    return vecs


def embed_jd(model: SentenceTransformer, out_path: Path, skills_path: Path) -> tuple[np.ndarray, np.ndarray]:
    jd_vec = model.encode(
        [JD_TEXT],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(out_path, jd_vec)
    print(f"  JD embedding saved: {out_path}  shape={jd_vec.shape}")

    skill_vecs = model.encode(
        JD_SKILLS,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(skills_path, skill_vecs)
    print(f"  JD skill matrix saved: {skills_path}  shape={skill_vecs.shape}")
    return jd_vec, skill_vecs
