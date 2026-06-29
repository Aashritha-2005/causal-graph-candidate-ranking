"""
FAISS index: build from candidate embeddings, query for shortlist.

At ranking time: load pre-built index from disk, query with JD embedding.
No model inference at ranking time — pure FAISS lookup.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray, out_path: Path) -> faiss.Index:
    """
    Build an exact inner-product index (IndexFlatIP).
    Embeddings must be unit-normalized (cosine similarity = inner product).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_path))
    print(f"  FAISS index saved: {out_path}  ({index.ntotal:,} vectors, dim={dim})")
    return index


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def retrieve_shortlist(
    index: faiss.Index,
    jd_embedding: np.ndarray,
    candidate_ids: list[str],
    k: int = 5000,
) -> pd.DataFrame:
    """
    Query FAISS index with JD embedding, return top-k candidates with cosine scores.

    Returns DataFrame: candidate_id, faiss_rank, cosine_sim
    """
    query = jd_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, k)
    scores = scores[0]
    indices = indices[0]

    # Filter out invalid indices (-1 can appear if k > ntotal)
    valid = indices >= 0
    indices = indices[valid]
    scores = scores[valid]

    return pd.DataFrame({
        "candidate_id": [candidate_ids[i] for i in indices],
        "faiss_rank": range(1, len(indices) + 1),
        "cosine_sim": scores.astype(float),
    })
