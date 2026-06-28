"""
Structural format tests — run against any submission CSV before upload.
These mirror the logic in validate_submission.py as a fast local check.

Usage:
  pytest tests/test_output_format.py --sub submission.csv
"""

import csv
import re
from pathlib import Path

import pytest

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
BUNDLE_DIR = Path(__file__).parent.parent / "India_runs_data_and_ai_challenge"


@pytest.fixture
def submission_path(request):
    path = request.config.getoption("--sub")
    if path is None:
        pytest.skip("Pass --sub path/to/submission.csv")
    return Path(path)


def load_rows(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["candidate_id", "rank", "score", "reasoning"], (
            f"Header mismatch: {reader.fieldnames}"
        )
        return list(reader)


def test_exactly_100_rows(submission_path):
    rows = load_rows(submission_path)
    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"


def test_ranks_1_to_100_each_once(submission_path):
    rows = load_rows(submission_path)
    ranks = [int(r["rank"]) for r in rows]
    assert sorted(ranks) == list(range(1, 101)), "Ranks must be 1–100 each exactly once"


def test_no_duplicate_candidate_ids(submission_path):
    rows = load_rows(submission_path)
    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate candidate_ids found"


def test_candidate_ids_valid_format(submission_path):
    rows = load_rows(submission_path)
    bad = [r["candidate_id"] for r in rows if not CANDIDATE_ID_RE.match(r["candidate_id"])]
    assert not bad, f"Invalid candidate_id formats: {bad[:5]}"


def test_scores_non_increasing(submission_path):
    rows = load_rows(submission_path)
    rows_by_rank = sorted(rows, key=lambda r: int(r["rank"]))
    scores = [float(r["score"]) for r in rows_by_rank]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"score not non-increasing at ranks {i+1},{i+2}: {scores[i]} < {scores[i+1]}"
        )


def test_score_tie_break_by_candidate_id(submission_path):
    rows = load_rows(submission_path)
    rows_by_rank = sorted(rows, key=lambda r: int(r["rank"]))
    for i in range(len(rows_by_rank) - 1):
        s1 = float(rows_by_rank[i]["score"])
        s2 = float(rows_by_rank[i + 1]["score"])
        c1 = rows_by_rank[i]["candidate_id"]
        c2 = rows_by_rank[i + 1]["candidate_id"]
        if s1 == s2:
            assert c1 < c2, (
                f"Tie at ranks {i+1},{i+2} (score={s1}): "
                f"candidate_id tie-break violated: {c1!r} > {c2!r}"
            )


def test_reasoning_not_empty(submission_path):
    rows = load_rows(submission_path)
    empty = [r["candidate_id"] for r in rows if not r.get("reasoning", "").strip()]
    assert not empty, f"Empty reasoning strings: {empty}"


def test_reasoning_not_all_identical(submission_path):
    rows = load_rows(submission_path)
    unique = len(set(r["reasoning"] for r in rows))
    assert unique >= 90, f"Only {unique} unique reasoning strings (too many duplicates)"


def test_candidate_ids_exist_in_dataset(submission_path):
    """Spot-check: submitted IDs must exist in candidates.jsonl (checks first 5K lines)."""
    candidates_path = BUNDLE_DIR / "candidates.jsonl"
    if not candidates_path.exists():
        pytest.skip("candidates.jsonl not in bundle dir")

    rows = load_rows(submission_path)
    submitted_ids = {r["candidate_id"] for r in rows}

    found = set()
    with open(candidates_path, encoding="utf-8") as f:
        import json
        for i, line in enumerate(f):
            if i >= 5000:
                break
            c = json.loads(line)
            if c["candidate_id"] in submitted_ids:
                found.add(c["candidate_id"])

    # Only ~5 of 100 IDs are expected in the first 5K lines by chance (uniform distribution).
    # This test catches format errors (e.g. wrong prefix) not coverage — pass if any were found.
    if len(found) == 0:
        pytest.fail("Zero submitted IDs found in first 5K lines — possible ID format error (check CAND_ prefix)")
