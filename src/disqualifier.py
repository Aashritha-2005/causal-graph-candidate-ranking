"""
Disqualifier-penalty scoring (formerly "causal debiasing" — renamed per Day 3-4 pre-work).

Applies soft penalties for JD disqualifier criteria using only structural features.
D1 and D2 dropped: fired on 0 candidates in this dataset (non-applicable).

Active flags (all structurally grounded):
  D3: no_prod_code_18mo   — senior arch/lead title, ≥18mo current role
  D4: frequent_job_hopper — >50% of roles < 18mo, ≥4 total roles (proxy for title-chasing)
  D5: low_github          — github_activity_score ≥ 0 but < 20
  D6: pure_services       — entire career at IT-services companies
  D7: cv_speech           — CV/speech/robotics title, no NLP/IR signal
  D8: closed_source       — no GitHub linked, low identity verification

Each penalty is a multiplicative factor in [0, 1].
Penalties compound: score × P_D3 × P_D4 × ... × P_D8.
These are soft penalties, not hard filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Penalty multipliers — how much to reduce score when flag is True
# Tuned to match JD's stated severity: D6 and D7 are strong disqualifiers,
# D4 and D5 are moderate, D8 is mild.
PENALTY = {
    "d3_no_prod_code":        0.80,   # mild: maybe still technically strong
    "d4_frequent_job_hopper": 0.85,   # mild: structural proxy, some false positives
    "d6_pure_services":       0.25,   # strong: JD explicitly rejects pure-services careers
    "d7_cv_speech":           0.50,   # strong: domain mismatch
    "d5_low_github":          0.90,   # mild: low public activity
    "d8_closed_source":       0.92,   # mild: weak signal
}


def compute_disqualifier_multiplier(df: pd.DataFrame) -> np.ndarray:
    """
    Compute compound penalty multiplier for each candidate.
    Returns array of shape (len(df),) with values in (0, 1].
    """
    multiplier = np.ones(len(df), dtype=np.float32)

    for flag, penalty in PENALTY.items():
        if flag in df.columns:
            mask = df[flag].values.astype(bool)
            multiplier[mask] *= penalty

    return multiplier
