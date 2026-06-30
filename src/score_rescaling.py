"""
Monotonic score rescaling via Platt sigmoid (Day 5).

This is NOT conformal prediction. It is Platt scaling (logistic sigmoid) fit on
self-referential pseudo-labels derived from the same composite score being rescaled.
The pseudo-labels (top-500 vs bottom-500 by composite score) are circular — fitting
a sigmoid to reproduce a score's own rank order will always produce a sigmoid shape.
No independent signal exists locally to calibrate against.

What this actually does:
- Preserves rank order exactly (monotone transform; zero impact on NDCG/MAP/P@10)
- Reshapes the score distribution for presentation: top candidates cluster near 1.0,
  tail candidates spread out more visibly
- Does NOT improve ranking quality in any measurable sense

Why it's kept:
- The reshaped distribution is a more honest representation of "distance to JD ideal"
  than the raw linear composite — high-confidence candidates visually separate from
  borderline ones
- Adds ~1.6s to the ranking pipeline (acceptable)

Assumption: build_feature_table is the sole writer of skills_json in the parquet.
(See PROJECT.md audit note — if that changes, re-verify downstream catches.)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _fit_platt_sigmoid(
    raw_scores: np.ndarray,
    top_k: int = 500,
    bottom_k: int = 500,
) -> LogisticRegression:
    """
    Fit logistic regression sigmoid on top-k vs bottom-k of shortlist scores.
    Labels are pseudo (derived from rank position, not ground truth).
    """
    n = len(raw_scores)
    top_k = min(top_k, n // 4)
    bottom_k = min(bottom_k, n // 4)

    X = np.concatenate([raw_scores[:top_k], raw_scores[n - bottom_k:]]).reshape(-1, 1)
    y = np.array([1] * top_k + [0] * bottom_k)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    clf.fit(X, y)
    return clf


def apply_platt_rescaling(
    all_raw_scores: np.ndarray,
    shortlist_indices: np.ndarray,
    top_k: int = 500,
    bottom_k: int = 500,
) -> np.ndarray:
    """
    Fit Platt sigmoid on shortlist, apply to all candidates.
    Candidates outside the shortlist get score 0.0.
    Rank order is preserved exactly — this is a presentation-only transform.

    Parameters
    ----------
    all_raw_scores    : (n_candidates,) raw composite scores, aligned to df index
    shortlist_indices : integer indices of shortlist candidates in all_raw_scores
    top_k, bottom_k   : pseudo-label counts (top/bottom of shortlist by raw score)

    Returns
    -------
    rescaled : (n_candidates,) float32 in [0, 1], zeros outside shortlist
    """
    shortlist_scores = all_raw_scores[shortlist_indices]
    sort_order = np.argsort(-shortlist_scores)
    sorted_scores = shortlist_scores[sort_order]

    sigmoid = _fit_platt_sigmoid(sorted_scores, top_k=top_k, bottom_k=bottom_k)

    rescaled = np.zeros(len(all_raw_scores), dtype=np.float32)
    rescaled[shortlist_indices] = sigmoid.predict_proba(
        shortlist_scores.reshape(-1, 1)
    )[:, 1].astype(np.float32)
    return rescaled
