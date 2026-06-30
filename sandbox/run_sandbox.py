#!/usr/bin/env python3
"""
Sandbox entry point: rank a small sample of candidates (≤100) end-to-end.

Usage (from repo root):
    python sandbox/run_sandbox.py \
        --candidates ./India_runs_data_and_ai_challenge/sample_candidates.json \
        --out ./sandbox_submission.csv

Or with a custom candidates file (JSON array or JSONL, ≤100 candidates):
    python sandbox/run_sandbox.py --candidates ./my_sample.jsonl --out ./out.csv

This bypasses precomputed artifacts — embeddings are computed on the fly for
the small sample. Same compute budget applies: CPU only, no network during ranking.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_candidates(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    # JSONL
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sandbox ranker — small sample, no precomputed artifacts")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    if len(candidates) > 100:
        print(f"  WARNING: {len(candidates)} candidates; truncating to 100 for sandbox")
        candidates = candidates[:100]
    top_n = min(args.top_n, len(candidates))
    print(f"  {len(candidates)} candidates loaded")

    # Parse features
    print("Parsing features...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.parse import parse_candidate
    rows = [parse_candidate(c) for c in candidates]
    df = pd.DataFrame(rows)

    # Embed on the fly (small sample — fast)
    print("Embedding candidates and JD...")
    from sentence_transformers import SentenceTransformer
    from src.embed import JD_TEXT, JD_SKILLS
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["embed_text"].tolist()
    cand_embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    jd_emb = model.encode([JD_TEXT], convert_to_numpy=True, normalize_embeddings=True)
    jd_skills = model.encode(JD_SKILLS, convert_to_numpy=True, normalize_embeddings=True)
    del model

    # Cosine similarities
    cosine_sim = (cand_embs @ jd_emb.T).squeeze().astype(np.float32)
    if cosine_sim.ndim == 0:
        cosine_sim = np.array([float(cosine_sim)], dtype=np.float32)

    # OT scores
    print("Computing OT scores...")
    from src.ot_matching import build_skill_embedding_cache, compute_ot_scores_from_skill_embeddings
    from sentence_transformers import SentenceTransformer as ST
    model2 = ST("all-MiniLM-L6-v2")
    skill_cache = build_skill_embedding_cache(df["skills_json"].tolist(), model2)
    del model2
    ot_scores = compute_ot_scores_from_skill_embeddings(
        df["skills_json"].tolist(), skill_cache, jd_skills
    )

    # Disqualifier penalties
    from src.disqualifier import compute_disqualifier_multiplier
    disq = compute_disqualifier_multiplier(df)

    # Score and rank
    print("Scoring and ranking...")
    from src.score_mvp import compute_scores, rank_candidates_from_calibrated
    raw = compute_scores(df, cosine_sim=cosine_sim, ot_scores=ot_scores, disq_multiplier=disq).values.astype(np.float32)
    ranked = rank_candidates_from_calibrated(df, raw, top_n=top_n)

    from src.reasoning import add_reasoning_column
    ranked = add_reasoning_column(ranked)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = ranked[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(args.out, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {args.out}")
    print(f"\nTop {min(5, top_n)}:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
