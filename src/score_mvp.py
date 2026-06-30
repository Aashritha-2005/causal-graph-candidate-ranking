"""
MVP scoring: weighted feature scoring without OT or disqualifier-penalty.
Produces a valid, submittable ranking for Days 1-2.

Score formula (Day 2+, with embeddings):
  score = (
      w_sem     * cosine_sim          # semantic similarity to JD (sentence-transformer)
    + w_career  * career_score        # product months, ML years, structured signals
    + w_avail   * availability_score  # redrob signals
    + w_loc     * location_score      # location fit
  ) * plausibility_score ^ PLAUS_EXP

Fallback (no embeddings): w_sem replaced by title+skill heuristic domain_score.
No per-candidate LLM calls at any point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- weights (Days 3-4 with OT + disqualifier penalties) ----------
W_SEM       = 0.35   # semantic cosine sim to JD (sentence-transformer)
W_OT        = 0.10   # OT/Sinkhorn skill distribution match (new Day 3-4)
W_CAREER    = 0.30   # product company months, ML/AI years, structured signals
W_AVAIL     = 0.15   # redrob availability signals
W_LOC       = 0.10   # location fit
PLAUS_EXP   = 0.40   # plausibility penalty exponent

# JD experience range
JD_YOE_MIN, JD_YOE_MAX = 5, 9

# ML/AI titles that are a strong positive signal for this JD
STRONG_ML_TITLES = {
    "machine learning engineer", "ml engineer", "senior ml engineer",
    "ai engineer", "senior ai engineer", "applied scientist",
    "data scientist", "senior data scientist", "nlp engineer",
    "research engineer", "research scientist", "llm engineer",
    "ml research engineer", "ai research engineer",
}

# Titles that are moderately relevant (data infra, software + ML exposure)
MODERATE_ML_TITLES = {
    "software engineer", "senior software engineer", "backend engineer",
    "full stack engineer", "data engineer", "analytics engineer",
    "ml ops engineer", "mlops engineer", "platform engineer",
}

# Skills that are strongly relevant to the JD
JD_CORE_SKILLS = {
    "embeddings", "sentence transformers", "faiss", "vector database", "vector db",
    "qdrant", "pinecone", "weaviate", "milvus", "opensearch", "elasticsearch",
    "retrieval", "ranking", "recommendation", "search", "information retrieval",
    "nlp", "natural language processing", "bert", "transformers", "hugging face",
    "pytorch", "tensorflow", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "ndcg", "map", "mrr", "a/b testing", "evaluation", "offline eval",
    "rag", "llm", "large language models", "gpt", "openai",
    "python", "machine learning", "deep learning", "neural network",
    "distributed systems", "mlflow", "weights & biases", "wandb",
    "spark", "kafka", "airflow", "data pipeline",
}


def _title_score(title: str) -> float:
    t = title.lower().strip()
    if t in STRONG_ML_TITLES:
        return 1.0
    if any(kw in t for kw in ("machine learning", "ml ", " ml", "ai engineer", "data scientist", "nlp", "llm")):
        return 0.9
    if t in MODERATE_ML_TITLES:
        return 0.5
    if any(kw in t for kw in ("software", "backend", "data", "engineer", "developer")):
        return 0.4
    return 0.1


def _skill_relevance_score(skills_json: str) -> float:
    """
    Weighted fraction of JD core skills present, weighted by proficiency × log(1+duration).
    """
    PROF_VAL = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    try:
        skills = json.loads(skills_json)
    except (json.JSONDecodeError, TypeError):
        # Only reachable if skills_json is null/empty for a candidate with no skills.
        # build_feature_table asserts all skills_json values are valid JSON, so a
        # decode error here means missing data, not a code bug.
        return 0.0

    total_weight = 0.0
    matched_weight = 0.0
    for s in skills:
        name = s.get("name", "").lower().strip()
        prof = PROF_VAL.get(s.get("proficiency", "beginner"), 1)
        dur = s.get("duration_months", 0)
        w = prof * math.log1p(dur)
        total_weight += w
        if any(jd_skill in name or name in jd_skill for jd_skill in JD_CORE_SKILLS):
            matched_weight += w

    if total_weight == 0:
        return 0.0
    return min(1.0, matched_weight / total_weight)


def _domain_score(row: pd.Series) -> float:
    title_s = _title_score(row["current_title"])
    skill_s = _skill_relevance_score(row["skills_json"])
    # 60% title (direct role fit), 40% skills (depth signal)
    return 0.60 * title_s + 0.40 * skill_s


def _career_score(row: pd.Series) -> float:
    """
    Structured career quality signals for AI Engineer fit.
    """
    score = 0.0

    # Years of experience fit (JD says 5-9, explicitly not a hard cutoff)
    yoe = row["yoe"]
    ml_months = row["ml_ai_months"]
    if JD_YOE_MIN <= yoe <= JD_YOE_MAX:
        yoe_score = 1.0
    elif yoe < JD_YOE_MIN:
        yoe_score = yoe / JD_YOE_MIN
    else:
        # Beyond 9yr: decay, but reduced if candidate has substantial ML months.
        # 80+ ml_ai_months means the "seniority" years are in-domain, not overhead.
        base_decay = max(0.5, 1.0 - (yoe - JD_YOE_MAX) * 0.03)
        ml_relief = min(0.15, (ml_months - 48) / 320)  # up to +0.15 for 96+ ml_months
        yoe_score = min(1.0, base_decay + max(0.0, ml_relief))
    score += 0.20 * yoe_score

    # Product company months (JD explicitly rejects pure-services careers)
    # Target: 4-5 years (48-60 months) at product companies
    prod_months = row["product_months"]
    prod_score = min(1.0, prod_months / 60)
    # Hard penalty for pure services career (D6)
    if row["d6_pure_services"]:
        prod_score *= 0.1
    score += 0.30 * prod_score

    # ML/AI months (applied AI/ML experience)
    ml_months = row["ml_ai_months"]
    ml_score = min(1.0, ml_months / 48)  # target: 4 years
    score += 0.30 * ml_score

    # Frequent job-hopper penalty (D4 renamed — structural, >50% roles <18mo)
    if row["d4_frequent_job_hopper"]:
        score *= 0.7

    # No-prod-code-18mo penalty (D3)
    if row["d3_no_prod_code"]:
        score *= 0.8

    # CV/speech/robotics without NLP penalty (D7)
    if row["d7_cv_speech"]:
        score *= 0.6

    # Current role in a relevant industry is a mild positive signal
    industry = str(row.get("current_industry", "")).lower()
    if any(kw in industry for kw in ("technology", "software", "fintech", "saas", "ai", "ml", "data")):
        score = min(1.0, score * 1.05)

    return min(1.0, score)


def compute_scores(
    df: pd.DataFrame,
    cosine_sim: np.ndarray | None = None,
    ot_scores: np.ndarray | None = None,
    disq_multiplier: np.ndarray | None = None,
) -> pd.Series:
    """
    Vectorized composite score (Days 3-4).

    cosine_sim      : (len(df),) float32 — cosine sim to JD embedding. If None, uses domain_score.
    ot_scores       : (len(df),) float32 — Sinkhorn OT score. If None, W_OT weight collapses to W_SEM.
    disq_multiplier : (len(df),) float32 — compound penalty in (0, 1]. If None, no penalty applied.
    """
    if cosine_sim is not None:
        sem = pd.Series(cosine_sim, index=df.index).clip(0, 1)
    else:
        sem = df.apply(_domain_score, axis=1)

    if ot_scores is not None:
        ot = pd.Series(ot_scores, index=df.index).clip(0, 1)
        w_sem_eff, w_ot_eff = W_SEM, W_OT
    else:
        ot = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        # redistribute OT weight to SEM when OT unavailable
        w_sem_eff, w_ot_eff = W_SEM + W_OT, 0.0

    career  = df.apply(_career_score, axis=1)
    avail   = df["availability_score"]
    loc     = df["location_score"]
    plaus   = df["plausibility_score"]

    raw = (
        w_sem_eff * sem
        + w_ot_eff * ot
        + W_CAREER  * career
        + W_AVAIL   * avail
        + W_LOC     * loc
    )

    score = raw * (plaus ** PLAUS_EXP)

    if disq_multiplier is not None:
        disq = pd.Series(disq_multiplier, index=df.index)
        score = score * disq

    return score.round(6)


def rank_candidates(
    df: pd.DataFrame,
    cosine_sim: np.ndarray | None = None,
    ot_scores: np.ndarray | None = None,
    disq_multiplier: np.ndarray | None = None,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Return top_n candidates sorted by score DESC, then candidate_id ASC for ties.
    Scores are normalized to [0, 1] range and made strictly non-increasing per rank.
    """
    df = df.copy()
    df["_raw_score"] = compute_scores(
        df, cosine_sim=cosine_sim, ot_scores=ot_scores, disq_multiplier=disq_multiplier
    )

    # Sort: score DESC, candidate_id ASC (tie-break per validator spec)
    df_sorted = df.sort_values(["_raw_score", "candidate_id"], ascending=[False, True])
    top = df_sorted.head(top_n).reset_index(drop=True)

    # Normalize scores to [0, 1] relative to this top-N
    max_s = top["_raw_score"].iloc[0]
    min_s = top["_raw_score"].iloc[-1]
    span = max_s - min_s if max_s > min_s else 1.0
    top["score"] = ((top["_raw_score"] - min_s) / span * 0.95 + 0.05).round(4)

    # Re-sort after rounding: score DESC, candidate_id ASC (validator tie-break rule)
    top = top.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    # Assign ranks 1..top_n
    top["rank"] = range(1, top_n + 1)

    # Enforce non-increasing score (rounding can still create tiny inversions after re-sort)
    for i in range(1, len(top)):
        if top.loc[i, "score"] > top.loc[i - 1, "score"]:
            top.loc[i, "score"] = top.loc[i - 1, "score"]

    return top


def rank_candidates_from_calibrated(
    df: pd.DataFrame,
    calibrated_scores: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Rank using pre-computed calibrated scores (output of conformal.py).
    Calibrated scores are already in [0, 1]; normalization stretches to [0.05, 1.0].
    """
    df = df.copy()
    df["_raw_score"] = calibrated_scores

    df_sorted = df.sort_values(["_raw_score", "candidate_id"], ascending=[False, True])
    top = df_sorted.head(top_n).reset_index(drop=True)

    # Normalize calibrated probs to [0.05, 1.0] for submission
    max_s = top["_raw_score"].iloc[0]
    min_s = top["_raw_score"].iloc[-1]
    span = max_s - min_s if max_s > min_s else 1.0
    top["score"] = ((top["_raw_score"] - min_s) / span * 0.95 + 0.05).round(4)

    top = top.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top["rank"] = range(1, top_n + 1)

    for i in range(1, len(top)):
        if top.loc[i, "score"] > top.loc[i - 1, "score"]:
            top.loc[i, "score"] = top.loc[i - 1, "score"]

    return top
