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

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
FEATURES_PATH   = ARTIFACTS_DIR / "structured_features.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "candidate_embeddings.npy"
JD_EMBED_PATH   = ARTIFACTS_DIR / "jd_embedding.npy"
INDEX_PATH      = ARTIFACTS_DIR / "faiss_index.bin"

SHORTLIST_K = 5000


def _ensure_features(candidates_path: Path):
    if not FEATURES_PATH.exists():
        print("[rank.py] structured_features.parquet not found — running parser now...")
        from src.parse import build_feature_table
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        build_feature_table(candidates_path, FEATURES_PATH, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Rank 100K candidates against Senior AI Engineer JD")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Load feature table
    # ------------------------------------------------------------------ #
    _ensure_features(args.candidates)
    print(f"[1/5] Loading feature table...")
    df = pd.read_parquet(FEATURES_PATH)
    candidate_ids = df["candidate_id"].tolist()
    print(f"      {len(df):,} candidates  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2. FAISS shortlist (embedding similarity to JD)
    # ------------------------------------------------------------------ #
    use_embeddings = EMBEDDINGS_PATH.exists() and JD_EMBED_PATH.exists() and INDEX_PATH.exists()
    cosine_sim_full = None

    if use_embeddings:
        print(f"[2/5] Loading embeddings and FAISS shortlist (k={SHORTLIST_K})...")
        from src.index import load_index, retrieve_shortlist

        jd_vec = np.load(JD_EMBED_PATH)
        index  = load_index(INDEX_PATH)

        shortlist_df = retrieve_shortlist(index, jd_vec, candidate_ids, k=SHORTLIST_K)

        # Build full cosine_sim array aligned to df (zeros for non-shortlisted)
        sim_map = dict(zip(shortlist_df["candidate_id"], shortlist_df["cosine_sim"]))
        cosine_sim_full = np.array([sim_map.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
        print(f"      Shortlist: {len(shortlist_df):,} candidates  ({time.time()-t0:.1f}s)")
    else:
        print(f"[2/5] Embedding artifacts not found — using title heuristic (fallback)")

    # ------------------------------------------------------------------ #
    # 3. Score and rank
    # ------------------------------------------------------------------ #
    print(f"[3/5] Scoring candidates...")
    from src.score_mvp import rank_candidates
    ranked = rank_candidates(df, cosine_sim=cosine_sim_full, top_n=args.top_n)
    print(f"      Top {args.top_n} selected  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 4. Template reasoning
    # ------------------------------------------------------------------ #
    print(f"[4/5] Generating reasoning...")
    from src.reasoning import add_reasoning_column
    ranked = add_reasoning_column(ranked)
    print(f"      Done  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 5. Write and validate
    # ------------------------------------------------------------------ #
    print(f"[5/5] Writing {args.out}...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = ranked[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(args.out, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {args.out}")
    print("\nTop 5:")
    print(out.head(5).to_string(index=False))

    print("\n--- Validator ---")
    from India_runs_data_and_ai_challenge.validate_submission import validate_submission
    errors = validate_submission(args.out)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Submission is valid.")


if __name__ == "__main__":
    main()
