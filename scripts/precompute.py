#!/usr/bin/env python3
"""
Offline precompute — run once before ranking step.
No time limit. Output saved to artifacts/.

Steps:
  1. Parse candidates.jsonl → structured_features.parquet  (if not present)
  2. Embed all candidates → candidate_embeddings.npy
  3. Embed JD text and JD skills → jd_embedding.npy, jd_skill_matrix.npy
  4. Build FAISS index → faiss_index.bin

Usage:
  python scripts/precompute.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl

Re-running is safe: each step checks if its output artifact already exists and
skips unless --force is passed.
"""

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
    )
    parser.add_argument("--force", action="store_true", help="Recompute all artifacts")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Step 1: Feature table
    # ------------------------------------------------------------------ #
    features_path = ARTIFACTS / "structured_features.parquet"
    if args.force or not features_path.exists():
        print("[1/4] Parsing candidates → feature table...")
        import sys; sys.path.insert(0, str(ROOT))
        from src.parse import build_feature_table
        df = build_feature_table(args.candidates, features_path, verbose=True)
    else:
        print(f"[1/4] Feature table exists — loading ({features_path})")
        import pandas as pd
        df = pd.read_parquet(features_path)
    print(f"      {len(df):,} candidates  ({time.time()-t0:.0f}s elapsed)\n")

    # ------------------------------------------------------------------ #
    # Step 2: Candidate embeddings
    # ------------------------------------------------------------------ #
    embeddings_path = ARTIFACTS / "candidate_embeddings.npy"
    if args.force or not embeddings_path.exists():
        print("[2/4] Embedding candidates...")
        import sys; sys.path.insert(0, str(ROOT))
        from src.embed import load_model, embed_candidates
        model = load_model()
        embeddings = embed_candidates(df, model, embeddings_path)
    else:
        print(f"[2/4] Candidate embeddings exist — loading ({embeddings_path})")
        import numpy as np
        embeddings = np.load(embeddings_path)
    print(f"      shape={embeddings.shape}  ({time.time()-t0:.0f}s elapsed)\n")

    # ------------------------------------------------------------------ #
    # Step 3: JD embeddings
    # ------------------------------------------------------------------ #
    jd_path = ARTIFACTS / "jd_embedding.npy"
    skills_path = ARTIFACTS / "jd_skill_matrix.npy"
    if args.force or not jd_path.exists() or not skills_path.exists():
        print("[3/4] Embedding JD and JD skills...")
        from src.embed import load_model, embed_jd
        # Reuse model if already loaded, else reload
        try:
            model
        except NameError:
            model = load_model()
        jd_vec, _ = embed_jd(model, jd_path, skills_path)
    else:
        print(f"[3/4] JD embeddings exist — loading")
        import numpy as np
        jd_vec = np.load(jd_path)
    print(f"      ({time.time()-t0:.0f}s elapsed)\n")

    # ------------------------------------------------------------------ #
    # Step 4: FAISS index
    # ------------------------------------------------------------------ #
    index_path = ARTIFACTS / "faiss_index.bin"
    if args.force or not index_path.exists():
        print("[4/4] Building FAISS index...")
        from src.index import build_index
        build_index(embeddings, index_path)
    else:
        print(f"[4/4] FAISS index exists ({index_path})")
    print(f"      ({time.time()-t0:.0f}s elapsed)\n")

    print(f"Precompute complete in {(time.time()-t0)/60:.1f} minutes.")
    print("Artifacts:")
    for p in sorted(ARTIFACTS.iterdir()):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:40s}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
