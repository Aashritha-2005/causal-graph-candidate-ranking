# Causal Graph Candidate Ranking

Submission for **Hack2Skill × Redrob AI — India.Runs Track 1: Intelligent Candidate
Discovery & Ranking Challenge**.

Ranks 100K candidates against a Senior AI Engineer job description, optimising
`0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10`.

---

## Reproduce

```bash
# Step 1: precompute artifacts (one-time; ~8 min; no time budget)
python scripts/precompute.py

# Step 2: rank (≤5 min / ≤16 GB RAM / CPU-only / zero network)
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
               --out ./submission.csv
```

Step 2 produces `submission.csv` in ~17 seconds on a 10-core CPU with 16 GB RAM.

---

## What was actually built

### Pipeline

```
candidates.jsonl
    │
    ▼
[parse.py]  Structured feature extraction (100K candidates → 45-col parquet)
    │        career_history → product_months, ml_ai_months, disqualifier flags
    │        skills → skills_json (ranked by proficiency × log duration)
    │        redrob_signals → availability_score, plausibility_score
    │
    ▼
[embed.py]  all-MiniLM-L6-v2 sentence-transformer embeddings
    │        candidate embed_text (headline | summary | career descs)
    │        JD text → 1×384 embedding; JD skills → 12×384 skill matrix
    │
    ▼
[index.py]  FAISS IndexFlatIP — cosine shortlist k=5000
    │
    ▼
[ot_matching.py]  Sinkhorn OT — candidate skill distribution vs JD 12-skill matrix
    │              Candidate skills weighted by proficiency × log(1+duration_months)
    │              Cost matrix: pairwise cosine distance between skill embeddings
    │              Skill embeddings built from all-MiniLM-L6-v2 at ranking time
    │
    ▼
[disqualifier.py]  Soft-penalty multiplier for JD disqualifier criteria
    │               D3: no recent prod code  D4: frequent job-hopper (with ML exemption)
    │               D5: low GitHub activity  D6: pure IT-services career
    │               D7: CV/speech/robotics   D8: no public code, low verification
    │               All flags structural — no free-text regex
    │
    ▼
[score_mvp.py]  Composite score
    │            W_SEM=0.35 (cosine sim) + W_OT=0.10 (Sinkhorn) + W_CAREER=0.30
    │            + W_AVAIL=0.15 + W_LOC=0.10, × plausibility^0.40 × disq_multiplier
    │
    ▼
[score_rescaling.py]  Platt sigmoid — monotonic score rescaling for presentation
    │                  Fit on shortlist pseudo-labels (top-500 vs bottom-500).
    │                  Preserves rank order exactly. Does NOT affect NDCG/MAP.
    │                  Reshapes score distribution: top candidates cluster near 1.0.
    │
    ▼
[reasoning.py]  Template-based reasoning — zero LLM, zero hallucination
                Pulls from real structured fields only (title, company, yoe,
                ml_ai_months, skills, availability, location)
```

### Naming corrections vs original ARCHITECTURE.md

| Original name | Actual name | Why renamed |
|---|---|---|
| Causal debiasing | Disqualifier-penalty scoring | Do-calculus not performed; logistic smoothing of heuristic flags |
| Conformal calibration | Platt sigmoid score rescaling | No held-out calibration set exists; pseudo-labels are circular; monotone transform, zero NDCG impact |

### What was not built

**Day 6 (GNN/heterogeneous-graph embeddings):** Cut per the roadmap's own stated
contingency ("first thing cut if behind schedule"). The real signal gaps it would have
addressed (company quality propagation, skill co-occurrence context) are marginal
improvements within an already-performing shortlist, not retrieval failures. Adding it
this late would have meant lower validation standards than every prior phase.

---

## Compute budget

| Step | What runs | Time | Network |
|---|---|---|---|
| `precompute.py` | Parse 100K → parquet, embed all candidates (all-MiniLM-L6-v2), embed JD, build FAISS index | ~8 min | none after model download |
| `rank.py` | Load parquet, FAISS shortlist k=5000, OT on shortlist, disqualifier penalties, composite score, Platt rescaling, reasoning, validate | **~17 s** | zero |

Model (`all-MiniLM-L6-v2`, 22 MB) is downloaded once by sentence-transformers to a
local cache. After that, `rank.py` makes zero network calls.

---

## Honeypot handling

`plausibility_score` is a continuous penalty, not a hard filter:

```
plausibility = 1.0 - (0.6 × date_penalty + 0.4 × proficiency_penalty)
date_penalty = excess_months / 120     # experience claimed > career timeline
proficiency_penalty = max(0, expert/advanced claimed, near-zero duration) / 24
```

Applied as `score × plausibility^0.40`. Detects ~43 of the spec's ~80 expected
honeypots via structured signals. Remaining ~37 gap is estimated (unverifiable locally —
no `is_honeypot` field in the dataset). The non-AI-title majority of honeypots are
naturally suppressed by domain scoring, not the plausibility term.

Verified after every scoring change: `python scripts/check_honeypots.py --sub submission.csv`

---

## Disqualifier flags

All flags are structurally grounded — no free-text regex used in any active flag.

| Flag | Signal source | Penalty |
|---|---|---|
| D1 (pure research) | Dropped — fires on 0/100K candidates | — |
| D2 (LLM-only, no pre-LLM ML) | Dropped — pattern absent from dataset | — |
| D3 (no prod code, 18+ mo) | Title + current role duration, structured | ×0.80 |
| D4 (frequent job-hopper) | >50% roles <18mo, ≥4 roles; **exempt if ml_ai_months≥36** | ×0.85 |
| D5 (low GitHub) | github_activity_score < 20, linked | ×0.90 |
| D6 (pure IT-services) | company name lookup, structured | ×0.25 |
| D7 (CV/speech/robotics) | current_title keyword, structured | ×0.50 |
| D8 (no public code) | github_score=-1 + low verification | ×0.92 |

D4 exemption: candidates with ≥36 months of applied ML/AI experience are not penalised
for frequent company switches — verified by profile inspection of CAND_0007412 (Applied
ML Engineer, career: LinkedIn → Glance → Swiggy → Locobuzz → Zoho, all ML roles).

---

## Tests and validation

```bash
# Output format tests (11 tests)
pytest tests/ --sub submission.csv -q

# Honeypot check
python scripts/check_honeypots.py --sub submission.csv

# Official validator
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

---

## Artifacts (not committed)

Precompute produces ~350 MB in `artifacts/`:

| File | Size | Description |
|---|---|---|
| `structured_features.parquet` | 42.7 MB | 100K × 45 feature columns |
| `candidate_embeddings.npy` | 153.6 MB | 100K × 384 float32 |
| `jd_embedding.npy` | <1 MB | 1 × 384 JD embedding |
| `jd_skill_matrix.npy` | <1 MB | 12 × 384 JD skill embeddings |
| `faiss_index.bin` | 153.6 MB | FAISS IndexFlatIP |

These are excluded from git (see `.gitignore`). Regenerate with `python scripts/precompute.py`.

---

## Dependencies

```
sentence-transformers==3.3.1
faiss-cpu==1.9.0
POT==0.9.4
pandas==2.2.3
numpy==1.26.4
pyarrow==17.0.0
scikit-learn==1.5.2
python-docx==1.1.2
pytest==8.3.4
```

Install: `pip install -r requirements.txt`
