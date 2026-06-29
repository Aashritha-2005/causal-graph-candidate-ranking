"""
Optimal Transport matching: candidate skill distribution vs JD skill distribution.

Each candidate is represented as a weighted distribution over their skill embeddings
(weights = proficiency_value × log(1 + duration_months), normalized to simplex).

The JD is represented as a uniform distribution over its required-skill embeddings
(precomputed in jd_skill_matrix.npy).

Sinkhorn distance = Earth Mover's Distance approximation between these two distributions.
OT score = 1 - normalized_distance (higher = better match).

CPU-feasible at k=5000 shortlist when batched.
No model inference at ranking time — all embeddings precomputed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import ot  # POT library

SINKHORN_REG = 0.05      # regularization (lower = closer to true EMD, slower)
SINKHORN_NUMITER = 100
MAX_SKILLS_PER_CANDIDATE = 20
BATCH_SIZE = 200          # candidates per batch for Sinkhorn

PROFICIENCY_VALUE = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def _candidate_skill_weights(skills_json: str) -> tuple[list[str], np.ndarray] | None:
    """
    Parse the pre-ranked skills_json field and return (skill_names, weights_simplex).
    Returns None if no usable skills.
    """
    try:
        skills = json.loads(skills_json)
    except (json.JSONDecodeError, TypeError):
        return None

    weights = []
    names = []
    for s in skills[:MAX_SKILLS_PER_CANDIDATE]:
        prof = PROFICIENCY_VALUE.get(s.get("proficiency", "beginner"), 1)
        dur = s.get("duration_months", 0)
        w = prof * math.log1p(dur)
        if w > 0:
            weights.append(w)
            names.append(s.get("name", ""))

    if not weights:
        return None

    w_arr = np.array(weights, dtype=np.float64)
    w_arr /= w_arr.sum()
    return names, w_arr


def _skill_name_to_embedding(
    skill_names: list[str],
    model,
) -> np.ndarray:
    """Embed a list of skill names using the sentence-transformer model."""
    vecs = model.encode(
        skill_names,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def compute_ot_scores_from_skill_embeddings(
    shortlist_skills_json: list[str],
    skill_name_to_vec: dict[str, np.ndarray],
    jd_skill_matrix: np.ndarray,
    jd_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute OT scores for a shortlist of candidates.

    Parameters
    ----------
    shortlist_skills_json : list of skills_json strings (one per candidate)
    skill_name_to_vec     : precomputed skill-name → embedding dict
    jd_skill_matrix       : (n_jd_skills, 384) — JD skill embeddings
    jd_weights            : (n_jd_skills,) — JD skill weights (uniform if None)

    Returns
    -------
    ot_scores : (n_candidates,) float array, OT score in [0, 1]
    """
    n_jd = jd_skill_matrix.shape[0]
    if jd_weights is None:
        jd_weights = np.ones(n_jd, dtype=np.float64) / n_jd

    ot_scores = np.zeros(len(shortlist_skills_json), dtype=np.float32)

    for i, skills_json in enumerate(shortlist_skills_json):
        result = _candidate_skill_weights(skills_json)
        if result is None:
            ot_scores[i] = 0.0
            continue

        skill_names, cand_weights = result

        # Build candidate skill embedding matrix
        cand_vecs = []
        valid_weights = []
        for name, w in zip(skill_names, cand_weights):
            vec = skill_name_to_vec.get(name.lower())
            if vec is not None:
                cand_vecs.append(vec)
                valid_weights.append(w)

        if not cand_vecs:
            ot_scores[i] = 0.0
            continue

        cand_matrix = np.stack(cand_vecs).astype(np.float64)  # (n_skills, 384)
        cand_w = np.array(valid_weights, dtype=np.float64)
        cand_w /= cand_w.sum()

        # Cost matrix: pairwise cosine distance (1 - cosine_sim)
        # Both matrices are unit-normalized so inner product = cosine similarity
        jd_mat = jd_skill_matrix.astype(np.float64)
        cosine_sim_mat = cand_matrix @ jd_mat.T  # (n_cand_skills, n_jd_skills)
        cost_matrix = np.clip(1.0 - cosine_sim_mat, 0, 2)  # distance ∈ [0, 2]

        # Sinkhorn distance
        try:
            dist = ot.sinkhorn2(
                cand_w, jd_weights,
                cost_matrix,
                reg=SINKHORN_REG,
                numItermax=SINKHORN_NUMITER,
                warn=False,
            )[0]
            # Normalize: max possible distance is 2.0 (orthogonal embeddings)
            ot_scores[i] = float(max(0.0, 1.0 - dist / 1.0))
        except Exception:
            ot_scores[i] = 0.0

    return ot_scores


def build_skill_embedding_cache(
    all_skills_json: list[str],
    model,
) -> dict[str, np.ndarray]:
    """
    Precompute embeddings for all unique skill names across all candidates.
    Called once per ranking session — saves re-embedding the same skill multiple times.
    """
    unique_skills: set[str] = set()
    for sj in all_skills_json:
        try:
            for s in json.loads(sj):
                name = s.get("name", "").strip().lower()
                if name:
                    unique_skills.add(name)
        except (json.JSONDecodeError, TypeError):
            pass

    skill_list = sorted(unique_skills)
    print(f"  Embedding {len(skill_list):,} unique skill names...")
    vecs = model.encode(
        skill_list,
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {name: vecs[i].astype(np.float32) for i, name in enumerate(skill_list)}
