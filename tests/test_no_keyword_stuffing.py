"""
Regression test: any submission produced by this system must NOT reproduce the
sample_submission.csv failure mode (non-AI-role candidates ranked highly on keyword count).

Run against sample_submission.csv on Day 1. Also parameterized to run against the
final submission.csv before upload.

Usage:
  pytest tests/test_no_keyword_stuffing.py                          # against sample
  pytest tests/test_no_keyword_stuffing.py --sub path/to/sub.csv   # against any csv
"""

import csv
import sys
from pathlib import Path

import pytest

BUNDLE_DIR = Path(__file__).parent.parent / "India_runs_data_and_ai_challenge"
SAMPLE_SUBMISSION = BUNDLE_DIR / "sample_submission.csv"

# Titles that must never dominate the top 10 of an AI Engineer ranking
NON_AI_TITLES = {
    "hr manager",
    "marketing manager",
    "content writer",
    "operations manager",
    "graphic designer",
    "sales executive",
    "accountant",
    "civil engineer",
    "mechanical engineer",
    "customer support",
    "project manager",
    "business analyst",
    "teacher",
    "finance manager",
    "administrative assistant",
}

TOP_N_STRICT = 10    # no non-AI title may appear here
TOP_N_LIMIT = 5      # max non-AI titles allowed in top 50


def load_submission(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: int(r["rank"]))
    return rows


def extract_title(reasoning: str) -> str:
    """Pull the first token group before 'with' from the reasoning field."""
    r = reasoning.strip()
    if " with " in r:
        return r.split(" with ")[0].strip().lower()
    return r.split()[0].lower() if r else ""


def _check_submission(path: Path):
    assert path.exists(), f"Submission file not found: {path}"
    rows = load_submission(path)
    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    violations_top10 = []
    violations_top50 = []

    for row in rows:
        rank = int(row["rank"])
        title = extract_title(row.get("reasoning", ""))
        is_non_ai = any(t in title for t in NON_AI_TITLES)

        if rank <= TOP_N_STRICT and is_non_ai:
            violations_top10.append((rank, row["candidate_id"], row.get("reasoning", "")[:60]))
        if rank <= 50 and is_non_ai:
            violations_top50.append((rank, row["candidate_id"]))

    assert not violations_top10, (
        f"Non-AI-role candidates found in top {TOP_N_STRICT} — this reproduces the "
        f"sample_submission.csv failure mode (keyword stuffing):\n"
        + "\n".join(f"  rank {r}: {cid} — {rsn}" for r, cid, rsn in violations_top10)
    )

    assert len(violations_top50) <= TOP_N_LIMIT, (
        f"Too many non-AI-role candidates in top 50 ({len(violations_top50)} > {TOP_N_LIMIT}). "
        f"Ranks: {[r for r, _ in violations_top50]}"
    )


def test_sample_submission_demonstrates_failure():
    """
    The sample_submission.csv MUST fail this test — it IS the failure mode.
    This test verifies the regression test is actually catching the problem.
    """
    path = SAMPLE_SUBMISSION
    if not path.exists():
        pytest.skip("sample_submission.csv not found in bundle dir")

    rows = load_submission(path)
    top10 = rows[:10]
    non_ai_in_top10 = [
        r for r in top10
        if any(t in extract_title(r.get("reasoning", "")) for t in NON_AI_TITLES)
    ]
    assert len(non_ai_in_top10) > 0, (
        "sample_submission.csv unexpectedly passed the keyword-stuffing check — "
        "verify that extract_title() is parsing the reasoning column correctly."
    )


def test_our_submission_does_not_keyword_stuff(submission_path):
    _check_submission(submission_path)


@pytest.fixture
def submission_path(request):
    path = request.config.getoption("--sub")
    if path is None:
        pytest.skip("Pass --sub path/to/submission.csv to run against a real submission")
    return Path(path)
