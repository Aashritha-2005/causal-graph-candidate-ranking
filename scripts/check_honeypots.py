#!/usr/bin/env python3
"""
Check honeypot position in a submission CSV.

Must be run after every scoring revision and result logged in PROJECT.md
under the "Honeypot position tracking" table.

Usage:
  python scripts/check_honeypots.py --sub submission.csv

Output:
  - Count of detectable honeypots (plausibility < 0.8) in top 100
  - Count of detectable honeypots in top 10
  - Honeypot rate (must stay ≤ 10% in top 100 to pass Stage 3)
  - Whether plausibility ≥ 0.8 for all top-100 candidates

Limitation: the ~35 undetectable honeypots (not caught by structured signals)
cannot be identified individually — we have no ground truth for them. A clean
result here does NOT guarantee they are absent from top 100.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
FEATURES = ROOT / "artifacts" / "structured_features.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, type=Path)
    args = parser.parse_args()

    if not FEATURES.exists():
        print("ERROR: artifacts/structured_features.parquet not found. Run parser first.")
        sys.exit(1)

    feats = pd.read_parquet(FEATURES, columns=["candidate_id", "plausibility_score",
                                                "current_title", "yoe"])
    sub = pd.read_csv(args.sub)

    merged = sub.merge(feats, on="candidate_id", how="left")
    merged_sorted = merged.sort_values("rank")

    top10 = merged_sorted[merged_sorted["rank"] <= 10]
    top100 = merged_sorted[merged_sorted["rank"] <= 100]

    # Detectable honeypots = plausibility < 0.8 (structured signal threshold)
    hp_top10 = top10[top10["plausibility_score"] < 0.8]
    hp_top100 = top100[top100["plausibility_score"] < 0.8]
    hp_rate_top100 = len(hp_top100) / len(top100)

    print(f"=== Honeypot check: {args.sub} ===")
    print(f"  Detectable honeypots in top 10:  {len(hp_top10)} / 10")
    print(f"  Detectable honeypots in top 100: {len(hp_top100)} / 100  (rate: {hp_rate_top100:.1%})")
    print(f"  Stage 3 limit is ≤10% — {'PASS' if hp_rate_top100 <= 0.10 else 'FAIL ❌'}")
    print(f"  All top-100 plausibility ≥ 0.8: {(top100['plausibility_score'] >= 0.8).all()}")

    if not hp_top100.empty:
        print("\n  Detectable honeypots found in top 100:")
        print(hp_top100[["rank", "candidate_id", "plausibility_score", "current_title", "yoe"]].to_string(index=False))

    print(f"\n  NOTE: ~35 undetectable honeypots cannot be checked here.")
    print(f"  Log this result in PROJECT.md honeypot tracking table.")


if __name__ == "__main__":
    main()
