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

ARTIFACTS_DIR   = Path("artifacts")
FEATURES_PATH   = ARTIFACTS_DIR / "structured_features.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "candidate_embeddings.npy"
JD_EMBED_PATH   = ARTIFACTS_DIR / "jd_embedding.npy"
JD_SKILLS_PATH  = ARTIFACTS_DIR / "jd_skill_matrix.npy"
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
    print(f"[1/6] Loading feature table...")
    df = pd.read_parquet(FEATURES_PATH)
    candidate_ids = df["candidate_id"].tolist()
    print(f"      {len(df):,} candidates  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2. FAISS shortlist
    # ------------------------------------------------------------------ #
    use_embeddings = all(p.exists() for p in [EMBEDDINGS_PATH, JD_EMBED_PATH, INDEX_PATH])
    cosine_sim_full = None
    shortlist_df = None

    if use_embeddings:
        print(f"[2/6] FAISS shortlist (k={SHORTLIST_K})...")
        from src.index import load_index, retrieve_shortlist
        jd_vec = np.load(JD_EMBED_PATH)
        index  = load_index(INDEX_PATH)
        shortlist_df = retrieve_shortlist(index, jd_vec, candidate_ids, k=SHORTLIST_K)

        sim_map = dict(zip(shortlist_df["candidate_id"], shortlist_df["cosine_sim"]))
        cosine_sim_full = np.array([sim_map.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
        print(f"      {len(shortlist_df):,} in shortlist  ({time.time()-t0:.1f}s)")
    else:
        print(f"[2/6] Embedding artifacts missing — using title heuristic fallback")

    # ------------------------------------------------------------------ #
    # 3. Optimal Transport matching (shortlist only)
    # ------------------------------------------------------------------ #
    ot_score_full = None
    if JD_SKILLS_PATH.exists() and shortlist_df is not None:
        print(f"[3/6] OT/Sinkhorn matching on shortlist...")
        from sentence_transformers import SentenceTransformer
        from src.ot_matching import build_skill_embedding_cache, compute_ot_scores_from_skill_embeddings

        jd_skills = np.load(JD_SKILLS_PATH)  # (12, 384)

        # Build skill embedding cache for shortlist candidates only
        shortlist_mask = df["candidate_id"].isin(set(shortlist_df["candidate_id"]))
        shortlist_feats = df[shortlist_mask].reset_index(drop=True)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        skill_cache = build_skill_embedding_cache(
            shortlist_feats["skills_json"].tolist(), model
        )
        del model  # free RAM after embedding

        ot_scores_shortlist = compute_ot_scores_from_skill_embeddings(
            shortlist_feats["skills_json"].tolist(),
            skill_cache,
            jd_skills,
        )

        # Map back to full candidate array
        ot_map = dict(zip(shortlist_feats["candidate_id"], ot_scores_shortlist))
        ot_score_full = np.array([ot_map.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
        print(f"      OT done  ({time.time()-t0:.1f}s)")
    else:
        print(f"[3/6] OT skipped (JD skill matrix or shortlist missing)")

    # ------------------------------------------------------------------ #
    # 4. Disqualifier-penalty multipliers
    # ------------------------------------------------------------------ #
    print(f"[4/6] Disqualifier penalties...")
    from src.disqualifier import compute_disqualifier_multiplier
    disq_mult = compute_disqualifier_multiplier(df)
    print(f"      Done  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 5. Raw composite scores + conformal calibration
    # ------------------------------------------------------------------ #
    print(f"[5/6] Scoring, calibration, and ranking...")
    from src.score_mvp import compute_scores, rank_candidates_from_calibrated

    candidate_ids_arr = np.array(candidate_ids)
    raw_scores = compute_scores(
        df,
        cosine_sim=cosine_sim_full,
        ot_scores=ot_score_full,
        disq_multiplier=disq_mult,
    ).values.astype(np.float32)

    if shortlist_df is not None:
        from src.conformal import calibrate_shortlist_scores
        # Build O(1) lookup instead of O(n×k) np.isin on string arrays
        id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
        shortlist_idx = np.array(
            [id_to_idx[cid] for cid in shortlist_df["candidate_id"] if cid in id_to_idx],
            dtype=np.intp,
        )
        calibrated_scores = calibrate_shortlist_scores(
            raw_scores, shortlist_idx, top_k=500, bottom_k=500
        )
        print(f"      Calibrated  ({time.time()-t0:.1f}s)")
    else:
        calibrated_scores = raw_scores

    ranked = rank_candidates_from_calibrated(df, calibrated_scores, top_n=args.top_n)

    from src.reasoning import add_reasoning_column
    ranked = add_reasoning_column(ranked)
    print(f"      Done  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 6. Write and validate
    # ------------------------------------------------------------------ #
    print(f"[6/6] Writing {args.out}...")
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
