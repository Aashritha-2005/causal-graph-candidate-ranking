#!/usr/bin/env python3
"""
rank.py — single entry point for the ranking step.

Usage:
  python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
                 --out ./submission.csv

Requires precomputed artifacts in ./artifacts/ (run scripts/precompute.py first).
Ranking step must complete in ≤5 min, ≤16 GB RAM, CPU only, zero network.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "structured_features.parquet"


def _ensure_artifacts(candidates_path: Path):
    """Auto-generate feature table if not present (first-run convenience)."""
    if not FEATURES_PATH.exists():
        print("[rank.py] structured_features.parquet not found — running parser now...")
        from src.parse import build_feature_table
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        build_feature_table(candidates_path, FEATURES_PATH, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Rank 100K candidates against Senior AI Engineer JD")
    parser.add_argument("--candidates", required=True, type=Path,
                        help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output CSV path (e.g. submission.csv)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of candidates to rank (default: 100)")
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Load feature table (precomputed artifact)
    # ------------------------------------------------------------------ #
    _ensure_artifacts(args.candidates)
    print(f"[1/4] Loading feature table from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"      {len(df):,} candidates loaded  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2. Score all candidates
    # ------------------------------------------------------------------ #
    print("[2/4] Scoring candidates (MVP weighted-feature model)...")
    from src.score_mvp import rank_candidates
    ranked = rank_candidates(df, top_n=args.top_n)
    print(f"      Top {args.top_n} selected  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 3. Generate template reasoning
    # ------------------------------------------------------------------ #
    print("[3/4] Generating reasoning strings...")
    from src.reasoning import add_reasoning_column
    ranked = add_reasoning_column(ranked)
    print(f"      Reasoning done  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 4. Write output CSV
    # ------------------------------------------------------------------ #
    print(f"[4/4] Writing {args.out}...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = ranked[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(args.out, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {args.out}")
    print("\nTop 5:")
    print(out.head(5).to_string(index=False))

    # Validate
    print("\n--- Running validator ---")
    from India_runs_data_and_ai_challenge.validate_submission import validate_submission
    errors = validate_submission(args.out)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Submission is valid.")


if __name__ == "__main__":
    main()
