# PROJECT.md — Causal Heterogeneous-Graph Candidate Ranking
> Single source of truth. Read in full before every code change.
> Append a changelog entry after every change.

---

## Goal
Rank the top 100 candidates from a 100K-candidate pool against a Senior AI Engineer job description (Redrob AI, Pune/Noida), maximizing:

```
0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

The system must beat naive keyword-matching by construction — that failure mode is explicitly tested in `tests/test_no_keyword_stuffing.py` on Day 1.

---

## Hard Constraints

| Constraint | Value |
|---|---|
| Ranking step wall-clock | ≤ 5 minutes |
| RAM | ≤ 16 GB |
| Compute | CPU only, no GPU |
| Disk for intermediate state | ≤ 5 GB |
| Network calls during ranking | Zero (no LLM APIs, no hosted models) |
| Output | `<participant_id>.csv`, UTF-8, 100 data rows + 1 header |
| Output columns | `candidate_id,rank,score,reasoning` (exact order) |
| Rank | integers 1–100, each exactly once |
| Score | float, monotonically non-increasing with rank |
| Score tie-break | secondary signal, else candidate_id ascending |
| Reasoning | 1-2 sentences, template-generated from real profile fields only |
| Honeypot rate in top 100 | ≤ 10% (else Stage 3 disqualification) |
| Submissions total | ≤ 3; last valid counts |

---

## Architecture (selected)

**Causal Heterogeneous-Graph Matching with Optimal Transport Ranking and Conformal Prediction**

Rationale: not a per-candidate LLM-scoring approach (infeasible under compute constraints and used by most competing teams). Not a repeat of prior work (GEPA/DSPy, RAG/LangGraph/ChromaDB, federated learning, neural twins).

See `ARCHITECTURE.md` for full design.

---

## Current Phase

**Day 1 — COMPLETE**
- [x] PROJECT.md created
- [x] ARCHITECTURE.md written and approved (with corrections before code)
- [x] Repo scaffolded (src/, tests/, requirements.txt, rank.py, conftest.py)
- [x] tests/test_no_keyword_stuffing.py written and passes (sample failure mode confirmed caught)
- [x] tests/test_output_format.py written — 9 structural checks, all pass
- [x] Parser (src/parse.py) — 47-column feature table over 100K candidates in 38s
- [x] MVP weighted scoring (src/score_mvp.py) — no OT/disqualifier-penalty yet
- [x] Template reasoning (src/reasoning.py) — zero LLM calls, zero hallucination
- [x] rank.py end-to-end: 8s wall-clock, validate_submission.py PASS
- [x] Top 100 quality check: all ML/AI titles, zero honeypots, zero IT-services-only, zero non-AI titles

**Day 2 — COMPLETE**
- [x] src/embed.py — sentence-transformer (all-MiniLM-L6-v2), JD text + skill matrix
- [x] src/index.py — FAISS IndexFlatIP, shortlist query
- [x] scripts/precompute.py — end-to-end offline precompute (8.6 min, 350 MB artifacts)
- [x] rank.py updated — loads FAISS, computes cosine_sim array, 2s ranking step
- [x] Honeypot re-verified: 0/43 in top 100; both ML-titled honeypots stayed out (cosine_sim 0.70 and 0.62, career+plausibility penalty sufficient)
- [x] 11/11 tests pass; validate_submission.py PASS

**Next: Days 3-4 — OT matching + disqualifier-penalty scoring**

---

## Phased Roadmap

| Phase | Days | Scope |
|---|---|---|
| MVP | 1-2 | Parse, feature table, embeddings, FAISS shortlist, weighted score, template reasoning, validator |
| Causal + OT | 3-4 | Propensity-style causal debiasing vs disqualifier list; Sinkhorn OT matching replacing simple weighted score |
| Conformal | 5 | Conformal calibration for confidence intervals; tune redrob_signals availability modifier; honeypot plausibility check |
| GNN (stretch) | 6 | Heterogeneous graph + GNN embeddings for retrieval — first thing cut if behind schedule |
| Deploy | 7 | Sandbox (HF Spaces / Streamlit Cloud / Docker), README repro command, submission_metadata.yaml, final validation, submit |

**Design principle:** every phase boundary must leave a fully working, submittable system. MVP (Days 1-2) does NOT depend on OT, causal, conformal, or GNN.

---

## Dataset Notes (verified against real files)

- `candidates.jsonl`: 100,000 lines, uncompressed (~465 MB). One JSON object per line.
- Format confirmed by reading actual file — not from prompt summary.

### Confirmed field structure

```
candidate_id: "CAND_XXXXXXX" (7 digits)
profile:
  anonymized_name, headline, summary, location, country
  years_of_experience (float), current_title, current_company
  current_company_size (enum), current_industry
career_history[]:
  company, title, start_date, end_date (nullable), duration_months
  is_current, industry, company_size, description
education[]:
  institution, degree, field_of_study, start_year, end_year, grade (nullable), tier
skills[]:
  name, proficiency (beginner/intermediate/advanced/expert)
  endorsements, duration_months
certifications[] (optional): name, issuer, year
languages[] (optional): language, proficiency
redrob_signals:
  profile_completeness_score (0-100)
  signup_date, last_active_date
  open_to_work_flag
  profile_views_received_30d, applications_submitted_30d
  recruiter_response_rate (0-1)
  avg_response_time_hours
  skill_assessment_scores (dict: skill→0-100)
  connection_count, endorsements_received
  notice_period_days, expected_salary_range_inr_lpa {min, max}
  preferred_work_mode (remote/hybrid/onsite/flexible)
  willing_to_relocate
  github_activity_score (-1 or 0-100; -1 = no GitHub linked)
  search_appearance_30d, saved_by_recruiters_30d
  interview_completion_rate (0-1), offer_acceptance_rate (-1 or 0-1)
  verified_email, verified_phone, linkedin_connected
```

### JD ideal-candidate calibration target (from actual JD)

- 6-8 years total, 4-5 years applied ML/AI at product companies (not pure services)
- Shipped end-to-end ranking/search/recommendation to real users at scale
- Production embeddings + hybrid retrieval experience
- Production vector DB experience
- Strong eval framework experience (NDCG, MRR, A/B)
- Located in / willing to relocate to Noida/Pune/Hyderabad/Mumbai/Delhi NCR

### JD hard disqualifiers (encode as penalty features)

1. Pure research background, no production deployment
2. AI experience < 12mo, LangChain-only, no pre-LLM-era ML
3. Senior engineer 18+ months with zero production code (pure arch/tech-lead)
4. Title-chaser: Senior→Staff→Principal via ~1.5yr hops
5. Framework enthusiast: GitHub of LangChain tutorials, no systems thinking
6. Entire career at IT services (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) with no product-company experience
7. CV/speech/robotics background, no NLP/IR exposure
8. 5+ years entirely closed-source, no external validation

### Honeypots (~80 candidates)
- Impossible experience math (e.g., 8 yrs experience at company founded 3 yrs ago)
- "expert" proficiency with near-zero duration_months
- Force to relevance tier 0 in hidden ground truth
- Honeypot rate in top 100 must stay ≤ 10% or disqualified at Stage 3
- Build `plausibility_score` as a real feature (not a hardcoded filter)

**Why Day 1 honeypots rank low — measured mechanism, not just a claim:**

`domain_score` for the 43 detectable honeypots: mean=0.225, median=0.137, max=0.833.
`domain_score` for the top 100 candidates: mean=0.816, median=0.817, min=0.707.

The separation is driven by `_title_score`: most honeypots have non-AI titles (HR Manager,
Marketing Manager, Accountant, Civil Engineer, etc.) which score 0.06–0.10. The highest-
scoring honeypots are the two with ML-adjacent titles (CAND_0019480 "NLP Engineer"
raw_score=0.698, CAND_0093547 "Senior ML Engineer" raw_score=0.669) — these are close
to the top-100 cut (min raw_score ≈ 0.780 after normalization). The plausibility penalty
(×0.4 exponent) suppresses them: 0.793^0.4 ≈ 0.913, 0.864^0.4 ≈ 0.943 — helpful but
not what's doing the heavy lifting. `domain_score` (title heuristic) is.

**Risk for Day 2:** semantic embeddings may score "NLP Engineer" honeypots differently
from the title heuristic — the JD text is rich in NLP/ML language, so a honeypot with
an ML title and fabricated skills could get a high cosine similarity even with low
plausibility. The two highest-scoring detectable honeypots are the exact ones to watch.

**Honeypot position tracking — must re-verify after every scoring revision:**

| Checkpoint | Scorer | Detectable in top 100 | Top-100 plaus ≥ 0.8? | Measured? | Notes |
|---|---|---|---|---|---|
| Day 1 MVP | title-heuristic domain_score | **0 / 43** | **Yes** (all = 1.0) | ✅ | Mechanism: title_score separation; highest-risk HP raw_scores 0.67–0.70 vs top-100 min ≈ 0.78 |
| Day 2 | embedding cosine sim (FAISS k=5000) | **0 / 43** | **Yes** (all = 1.0) | ✅ | Highest-risk HPs: CAND_0093547 cosine_sim=0.70 (stayed out — career+plaus penalty sufficient); CAND_0019480 cosine_sim=0.62 (well clear) |
| Days 3-4 | OT + disqualifier-penalty | — | — | ❌ | Run again |
| Day 5 | + conformal calibration | — | — | ❌ | Run again |

Re-verification command: `python scripts/check_honeypots.py --sub submission.csv`

**On the ~35 "undetectable" gap — no ground truth available locally:**

The ~80 figure comes solely from `submission_spec.docx` Section 7: *"a small number (~80)
of honeypot candidates."* No count, no list, no labeled field exists in the bundle — there
is no `is_honeypot` field in the schema, no separate file, no validator output. The hidden
ground truth is server-side only.

Structured signals catch 43. The remaining gap (~37) is an estimate derived by subtraction
from an approximate spec figure. It is **not a measured count**. The tracking table entry
"~35 undetectable" should be read as: *estimated gap, true count and identities
unverifiable locally*. A clean `plausibility ≥ 0.8` check covers only what structured
signals can see.

---

## Known Risks & Open Questions

1. **Embedding precompute time**: sentence-transformers on 100K summaries+descriptions — need to test throughput. If slow, consider batch size tuning or `all-MiniLM-L6-v2` (fastest). Precompute has no time limit; ranking step does.
2. **FAISS shortlist size**: need to determine optimal shortlist size k (e.g., 1000-5000) to feed into OT/scoring. Too small → miss real candidates. Too large → OT cost grows.
3. **OT matrix dimension**: Sinkhorn on shortlist × JD skill-set. Need to benchmark cost at k=1000.
4. **Participant ID for output filename**: not yet known — use `submission.csv` locally, rename before upload.
5. **Conformal calibration reference set**: no labeled calibration set provided. Will use held-out shortlist candidates ranked by domain heuristics as pseudo-labels.
6. **GNN**: stretch goal — first thing cut if behind schedule on Day 6.

---

## Assumptions (to verify and correct if wrong)

- [x] Dataset is 100,000 lines — **confirmed** (`wc -l` = 100000)
- [x] Field names/types match schema — **confirmed** (spot-checked CAND_0000001-5)
- [x] `duration_months` present in skills — **confirmed**
- [x] `github_activity_score` = -1 when no GitHub linked — **confirmed**
- [x] `offer_acceptance_rate` = -1 when no offer history — **confirmed against real data** (59,554 candidates have oar=-1; rest have float values 0–1; schema-derived trust upgraded to data-verified)
- [x] All candidates have at least one career_history entry — **confirmed** (scanned all 100K; zero candidates with empty career_history)
- [x] Honeypots are detectable via date math / proficiency-duration mismatch — **partially confirmed, important nuance:**
  - Date-math signal (sum_months > yoe×12 + 36mo): **22 candidates** flagged. Examples: CAND_0040075 declared 15yr but sum_months=365 (30.4yr, +15.4yr gap).
  - Proficiency-duration signal (expert skill with 0 months): **21 candidates** flagged. Examples: CAND_0003582 claims 'MLflow' expert at 0 months.
  - No overlap between the two signals (union = 43).
  - Widening thresholds (diff>24mo OR expert<6mo OR advanced<3mo) reaches only 46.
  - **Expected ~80 honeypots per spec; only ~43-46 detectable via structured fields.** Remaining ~35 likely embedded in free-text description inconsistencies or require external company-founding-date knowledge (not in dataset). Plausibility score will catch the ~43-46 clear cases; the rest will naturally score low on domain-relevance features without special-casing.

---

## Running Changelog

### 2026-06-28 — Day 1 complete
- Full MVP pipeline working end-to-end: parse → score → reasoning → CSV
- rank.py: 8s wall-clock (5min budget). validate_submission.py PASS. 11/11 tests pass.
- Top 100: all ML/AI professionals, zero IT-services-only, zero honeypots, plausibility all ≥ 0.8
- MVP domain_score is title+skill heuristic (no embeddings yet). Day 2 adds sentence-transformer cosine sim to replace/augment domain_score.
- Commit df4a0b6 (initial) + format test fix added.
- What's still open: embedding precompute (scripts/precompute.py + src/embed.py + src/index.py), FAISS shortlist integration into scorer.

### 2026-06-28 — Pre-Day-1 Verifications (post-approval)
- Ran all three data verifications required before scoring code.
- career_history: all 100K candidates have ≥1 entry — assumption confirmed.
- offer_acceptance_rate: upgraded from schema-derived to data-verified (59,554 = -1, rest are 0–1 floats).
- Honeypot detectability: ~43-46 catchable via structured signals (date-math + proficiency×duration); remaining ~35 likely require text or external knowledge. plausibility_score design is sound; won't catch all ~80 but catches the most egregious and naturally-avoids others via low domain-relevance scores.
- Updated ARCHITECTURE.md: (1) renamed "causal debiasing" to "disqualifier-penalty scoring" to avoid overclaiming; (2) added Known Risk on D1/D2/D4 text-pattern dependency before Days 3-4.
- What's still open: Day 1 code (test + parser + MVP scorer).

### 2026-06-28 — Step 0
- Created PROJECT.md after reading all bundle files (docx via python-docx, schema, sample candidates, sample submission, validator)
- Confirmed dataset structure directly from candidates.jsonl (not from prompt summary)
- Key findings:
  - `candidates.jsonl` is 100K lines uncompressed (NOT gzipped in the bundle — .gz not present)
  - `sample_submission.csv` confirms the failure mode: HR Manager ranked #1 on AI-skill keyword count
  - Validator checks: 100 rows, ranks 1-100 each once, score non-increasing, tie-break by candidate_id ascending
  - Submission_spec confirms: no LLM at ranking, 5min/16GB/CPU budget, sandbox required
- What's still open: ARCHITECTURE.md approval before any code
