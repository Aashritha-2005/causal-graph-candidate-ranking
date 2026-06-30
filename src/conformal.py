"""
Conformal calibration for ranking scores (Day 5).

Approach: Platt scaling (logistic sigmoid) fit on pseudo-labels from the shortlist.
- Pseudo-positive: top-500 candidates in shortlist by composite score (likely relevant)
- Pseudo-negative: bottom-500 candidates in shortlist (likely not relevant)
- Fit a logistic regression mapping raw composite score → calibrated probability
- Apply to all top-100 candidates to produce well-spread calibrated scores

This is a monotone score transformation (preserves rank order) but reshapes the
distribution so the score spread in the top-10 region is wider, which helps
NDCG@10 by making genuine quality differences more visible.

No external labels, no LLM calls. Calibration runs on the shortlist raw scores only.

Conformal interpretation: the logistic output approximates P(candidate is JD-relevant),
calibrated against the shortlist's own score distribution as a proxy for relevance.
Coverage guarantee is informal (pseudo-labels, not ground truth) — the benefit is
better score spread, not formal uncertainty quantification.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_calibrator(
    raw_scores: np.ndarray,
    top_k: int = 500,
    bottom_k: int = 500,
) -> LogisticRegression:
    """
    Fit a Platt scaling sigmoid on shortlist pseudo-labels.

    Parameters
    ----------
    raw_scores : (n_shortlist,) float array, composite scores before normalization,
                 sorted descending (shortlist order).
    top_k      : how many top candidates to label pseudo-positive (1)
    bottom_k   : how many bottom candidates to label pseudo-negative (0)

    Returns
    -------
    Fitted LogisticRegression that maps raw score → calibrated probability.
    """
    n = len(raw_scores)
    top_k = min(top_k, n // 4)
    bottom_k = min(bottom_k, n // 4)

    pos_scores = raw_scores[:top_k]
    neg_scores = raw_scores[n - bottom_k:]

    X = np.concatenate([pos_scores, neg_scores]).reshape(-1, 1)
    y = np.array([1] * top_k + [0] * bottom_k)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    clf.fit(X, y)
    return clf


def apply_calibration(
    calibrator: LogisticRegression,
    raw_scores: np.ndarray,
) -> np.ndarray:
    """
    Map raw composite scores to calibrated probabilities in [0, 1].
    Preserves relative order (monotone by construction if logistic is monotone).
    """
    probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    return probs.astype(np.float32)


def calibrate_shortlist_scores(
    all_raw_scores: np.ndarray,
    shortlist_indices: np.ndarray,
    top_k: int = 500,
    bottom_k: int = 500,
) -> np.ndarray:
    """
    Full pipeline: fit calibrator on shortlist, apply to all candidates.

    Parameters
    ----------
    all_raw_scores    : (n_candidates,) raw composite scores, aligned to df
    shortlist_indices : indices into all_raw_scores for the shortlist
    top_k, bottom_k   : pseudo-label counts

    Returns
    -------
    calibrated_scores : (n_candidates,) probabilities in [0, 1]
                        zeros for candidates not in shortlist
    """
    shortlist_scores = all_raw_scores[shortlist_indices]
    # Sort descending to define top_k / bottom_k
    sort_order = np.argsort(-shortlist_scores)
    sorted_scores = shortlist_scores[sort_order]

    calibrator = fit_platt_calibrator(sorted_scores, top_k=top_k, bottom_k=bottom_k)

    calibrated = np.zeros(len(all_raw_scores), dtype=np.float32)
    calibrated[shortlist_indices] = apply_calibration(calibrator, shortlist_scores)
    return calibrated
