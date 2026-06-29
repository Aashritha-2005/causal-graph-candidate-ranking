# ARCHITECTURE.md — Causal Heterogeneous-Graph Candidate Ranking

> **No code is written until this document is explicitly approved.**

---

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PRECOMPUTE (no time limit — run once, save artifacts)                     │
│                                                                            │
│  candidates.jsonl                                                          │
│       │                                                                    │
│       ▼                                                                    │
│  [A] PARSE + FEATURE TABLE ────────────────────────────────────────────┐  │
│       │  structured_features.parquet                                    │  │
│       │  (career dates, skill proficiency×duration, company type,      │  │
│       │   honeypot plausibility, disqualifier flags, redrob signals)   │  │
│       ▼                                                                 │  │
│  [B] TEXT EMBEDDING                                                     │  │
│       │  sentence-transformer (all-MiniLM-L6-v2)                       │  │
│       │  Embed: profile summary + career descriptions (concatenated)   │  │
│       │  candidate_embeddings.npy  (100K × 384, float32 → ~147 MB)    │  │
│       ▼                                                                 │  │
│  [C] FAISS INDEX                                                        │  │
│       │  faiss_index.bin  (IVFFlat or Flat, inner-product)             │  │
│       ▼                                                                 │  │
│  [D] JD ENCODING                                                        │  │
│       │  Embed JD text (same model)                                    │  │
│       │  jd_embedding.npy  (1 × 384)                                   │  │
│       │  jd_skill_matrix.npy  (n_jd_skills × 384)                     │  │
│       │  (JD parsed into skill-concept vectors for OT matching)        │  │
│       │                                                                 │  │
│  [E] CONFORMAL CALIBRATION (Day 5)                                      │  │
│       │  Fit nonconformity thresholds on pseudo-labeled holdout         │  │
│       │  conformal_thresholds.pkl                                       │  │
│       │                                                                 │  │
│  [F] GRAPH EMBEDDINGS (Day 6 stretch only)                              │  │
│       │  Build bipartite candidate–skill graph                          │  │
│       │  Run node2vec / lightweight GNN                                 │  │
│       │  graph_embeddings.npy                                           │  │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  RANKING STEP  (≤ 5 min, ≤ 16 GB RAM, CPU only, zero network)            │
│                                                                            │
│  Load artifacts                                                            │
│       │                                                                    │
│       ▼                                                                    │
│  [1] FAISS RETRIEVAL SHORTLIST                                            │
│       │  Query: jd_embedding                                              │
│       │  k = 5000 candidates retrieved (tunable)                         │
│       │  Output: shortlist of candidate indices + cosine sim scores       │
│       ▼                                                                    │
│  [2] PLAUSIBILITY / HONEYPOT FILTER                                       │
│       │  For each shortlist candidate, compute plausibility_score:        │
│       │  - career date consistency (sum of duration_months vs             │
│       │    end_date - start_date vs years_of_experience)                  │
│       │  - skill proficiency vs duration_months consistency               │
│       │    (expert with 0-3 months → honeypot signal)                    │
│       │  - company founding date vs claimed tenure (heuristic)           │
│       │  plausibility_score ∈ [0, 1]; candidates with score < 0.3        │
│       │  get a heavy penalty (not a hard filter — stays differentiable)  │
│       ▼                                                                    │
│  [3] DISQUALIFIER-PENALTY SCORING (Days 3-4)                             │
│       │  For each shortlist candidate, compute 8 disqualifier penalties:  │
│       │  [D1 DROPPED: 0 detections in dataset — no research/academia      │
│       │   industry exists; text heuristic fires on nobody. Non-applicable]│
│       │  [D2 DROPPED: 0 detections — only 64 candidates mention LLM      │
│       │   keywords at all; none without pre-LLM hits. Non-applicable]    │
│       │  D3: no_prod_code_18mo (senior title, last 18mo = arch only)     │
│       │  D4→frequent_job_hopper (>50% roles <18mo, ≥4 roles — STRUCTURAL)│
│       │     Proxy for title-chasing; 1,382 flagged, 9 are ML/AI titles   │
│       │  D5: framework_enthusiast (github_activity_score low)            │
│       │  D6: pure_services_flag (all career = TCS/Infosys/Wipro/... )    │
│       │  D7: cv_speech_robotics_no_nlp (primary domain mismatch)         │
│       │  D8: closed_source_only_5yr (no github + large enterprise only)  │
│       │                                                                    │
│       │  Disqualifier-penalty smoothing (NOT "causal debiasing"):        │
│       │  - Logistic regression fits P(flag=1 | structured features)      │
│       │  - Score multiplied by product of (1 - P(Di)) for each Di        │
│       │  - This smooths heuristic binary flags into soft penalties        │
│       │  - Does NOT perform do-calculus or adjust for confounders         │
│       │  - README will describe this honestly as "soft penalty scoring"  │
│       ▼                                                                    │
│  [4] OPTIMAL TRANSPORT MATCHING (Days 3-4)                               │
│       │  Represent each candidate as a distribution over skill vectors:   │
│       │  - candidate skill matrix C_i ∈ R^{n_skills × 384}              │
│       │  - weights = proficiency × log(1 + duration_months)             │
│       │  - normalized to probability simplex                             │
│       │                                                                    │
│       │  Represent JD as distribution over required-skill vectors:        │
│       │  - jd_skill_matrix J ∈ R^{n_jd_skills × 384}                   │
│       │  - weights = priority inferred from JD text position             │
│       │                                                                    │
│       │  OT distance = Sinkhorn(C_i, J, ε=0.1)                         │
│       │  (Earth Mover's Distance approximation, regularized)             │
│       │  OT_score = 1 - normalized_OT_distance                          │
│       │                                                                    │
│       │  Implementation: POT (Python Optimal Transport) sinkhorn_log    │
│       │  Vectorized over shortlist — feasible on CPU at k=5000          │
│       ▼                                                                    │
│  [5] AVAILABILITY MODIFIER (redrob_signals)                              │
│       │  availability_score = weighted combination of:                   │
│       │  - open_to_work_flag (binary, high weight)                       │
│       │  - days_since_last_active (recency decay, exponential)           │
│       │  - recruiter_response_rate                                       │
│       │  - notice_period_days (lower = better for hiring velocity)       │
│       │  - willing_to_relocate OR India location match                   │
│       │  - interview_completion_rate                                     │
│       │                                                                    │
│       │  availability_score ∈ [0, 1]                                     │
│       │  Applied as multiplicative modifier: final × availability^0.3   │
│       │  (exponent tuned: don't fully kill a great candidate, but        │
│       │   meaningfully penalize unavailable ones)                        │
│       ▼                                                                    │
│  [6] COMPOSITE SCORE                                                      │
│       │                                                                    │
│       │  MVP (Days 1-2, no OT/causal):                                  │
│       │    score = w1×cosine_sim + w2×career_feature + w3×availability  │
│       │                                                                    │
│       │  Full (Days 3-4+):                                               │
│       │    score = (α×OT_score + β×cosine_sim + γ×career_feature)       │
│       │             × (1 - P(disqualified))                              │
│       │             × availability_score^δ                               │
│       │             × plausibility_score^η                               │
│       │                                                                    │
│       │  Weights (α,β,γ,δ,η) tuned on domain reasoning, not grid search  │
│       │  Initial: α=0.35, β=0.25, γ=0.25, δ=0.3, η=0.15               │
│       ▼                                                                    │
│  [7] CONFORMAL CONFIDENCE INTERVAL (Day 5)                               │
│       │  Nonconformity score: distance from candidate score to JD        │
│       │  calibrated threshold at each rank                               │
│       │  Produces: confidence_lower, confidence_upper per candidate      │
│       │  Used in: reasoning generation ("strong fit", "likely fit",      │
│       │           "marginal fit" based on CI position)                   │
│       ▼                                                                    │
│  [8] DETERMINISTIC RANK ASSIGNMENT                                       │
│       │  Sort by score DESC                                              │
│       │  Tie-break: secondary signal (career_feature) DESC, then        │
│       │             candidate_id ASC                                     │
│       │  Take top 100                                                    │
│       ▼                                                                    │
│  [9] TEMPLATE REASONING GENERATION                                       │
│       │  For each of top 100 candidates, construct reasoning string:     │
│       │  Template pulls from REAL profile fields only:                   │
│       │  - current_title, years_of_experience, current_company           │
│       │  - strongest 2-3 relevant skills (by proficiency×duration)       │
│       │  - career highlight (product company + shipped system signal)    │
│       │  - gaps/concerns (if disqualifier partially triggered)           │
│       │  - availability signal (if notable)                              │
│       │  Zero LLM calls. Zero fabricated facts.                          │
│       ▼                                                                    │
│  OUTPUT: submission.csv                                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Designs

### A. Parse + Feature Table

One pass over `candidates.jsonl` (streaming, line-by-line) → pandas DataFrame saved as parquet.

**Features extracted per candidate:**

| Feature | Description | Type |
|---|---|---|
| `yoe` | years_of_experience from profile | float |
| `career_months_total` | sum of duration_months across all roles | int |
| `yoe_consistency` | abs(yoe - career_months_total/12) / yoe | float (honeypot) |
| `max_skill_proficiency_inconsistency` | max over skills of (proficiency_num - f(duration_months)) | float (honeypot) |
| `india_location` | country == "India" OR willing_to_relocate | bool |
| `preferred_city_match` | location ∈ {Pune, Noida, Hyderabad, Mumbai, Delhi, Bengaluru} | bool |
| `product_company_months` | months at non-IT-services companies | int |
| `services_only` | all career at {TCS,Infosys,Wipro,Accenture,Cognizant,Capgemini,...} | bool |
| `has_ranking_search_rec` | career descriptions mention ranking/search/recommendation | bool |
| `has_embeddings_vector_db` | career descriptions mention embeddings/vector DB/FAISS/etc | bool |
| `applied_ml_years` | estimated years in ML/AI roles (title heuristic) | float |
| `pure_research_flag` | D1 disqualifier | bool |
| `llm_only_ai_flag` | D2 disqualifier | bool |
| `title_chaser_flag` | D4 disqualifier | bool |
| `services_career_flag` | D6 disqualifier | bool |
| `notice_period_days` | from redrob_signals | int |
| `days_since_active` | (today - last_active_date).days | int |
| `recruiter_response_rate` | from redrob_signals | float |
| `open_to_work` | from redrob_signals | bool |
| `github_score` | from redrob_signals (-1 → 0) | float |
| `interview_completion_rate` | from redrob_signals | float |
| `plausibility_score` | composite honeypot detection score | float [0,1] |

### B. Text Embedding

Model: `all-MiniLM-L6-v2` (22M params, 384-dim, fast on CPU, well-suited for semantic similarity)

Text per candidate (concatenated, truncated to 512 tokens):
```
{headline} | {summary} | {career_description_1} | {career_description_2}
```

Batch size: 256 (tune for RAM), estimated throughput ~1000 candidates/sec on CPU → ~100 seconds for 100K.
Save as `artifacts/candidate_embeddings.npy` (100K × 384 float32 = ~147 MB).

JD text: parse from `job_description.docx`. Embed full JD text → `artifacts/jd_embedding.npy`.
JD skill extraction: regex + keyword matching on JD "Things you absolutely need" section → list of skill phrases → embed each → `artifacts/jd_skill_matrix.npy`.

### C. FAISS Index

Index type: `IndexFlatIP` (exact inner-product, ~147 MB in RAM, fast enough for k=5000 query at 100K scale).
Fallback: `IndexIVFFlat` with nlist=256 if exact is too slow (approximate, faster).
Save as `artifacts/faiss_index.bin`.

Shortlist size k=5000: generous enough to include all real candidates, small enough for OT to be fast.

### D. Optimal Transport (Sinkhorn)

Each candidate → skill distribution:
- Take up to 20 skills with proficiency ∈ {intermediate, advanced, expert}
- Weight = proficiency_value × log(1 + duration_months) where proficiency_value: beginner=1, intermediate=2, advanced=3, expert=4
- Embed skill name → 384-dim vector (using same model)
- Normalize weights to simplex → skill distribution μ_i ∈ Δ^{n_i}

JD → skill distribution:
- Parse required skills from JD → embed → normalize → ν ∈ Δ^{n_jd}

Cost matrix M: pairwise cosine distances between candidate-skill vectors and JD-skill vectors.

Sinkhorn distance (POT library, `ot.sinkhorn2`):
- Regularization ε = 0.05 (tighter for precision)
- Max iterations: 50
- Vectorized across shortlist in batches of 100

OT score = 1 - sinkhorn_distance / max_possible_distance (normalized to [0,1])

### E. Disqualifier-Penalty Scoring

Problem: raw keyword presence is misleading. A candidate with many AI keywords may still be disqualified by D1-D8 criteria from the JD.

Method:
1. Compute 8 binary disqualifier flags (D1-D8) as structured features (see below for signal sources)
2. Fit lightweight logistic regression: P(flag=1 | structured features) — smooths hard flags into soft penalties
3. Final score multiplied by product of (1 - P(Di)) for each Di

Signal sources per flag:
- **D1 DROPPED**: text heuristic fires on 0 candidates; no "research/academia" industry exists in this dataset. Non-applicable to this dataset.
- **D2 DROPPED**: 64/100K candidates mention any LangChain keyword; zero have LLM hits without pre-LLM hits. Non-applicable.
- **D3** (no prod code 18mo): current role title pattern (arch/lead/principal) + duration_months in current role. Structured.
- **D4 → `frequent_job_hopper`**: >50% of roles have duration_months < 18 AND ≥ 4 total roles. Purely structural (duration_months field). 1,382 flagged: 1,373 non-AI titles (already low-scoring), 9 ML/AI titles (minor false-positive risk). This is a proxy for the JD's title-chaser concern — it detects frequent switching, not title elevation specifically. Documented as proxy.
- **D5** (framework enthusiast): github_activity_score + career description keyword density. Text-dependent, heuristic.
- **D6** (pure services): check all companies against known IT-services list. Structured, reliable.
- **D7** (CV/speech/robotics): current_title + career history industry. Structured, reliable.
- **D8** (closed-source only): github_activity_score = -1 AND all companies are large enterprise. Heuristic.

**Naming note:** This is "disqualifier-penalty smoothing," not causal debiasing. There is no do-calculus, no defined treatment variable, no confounder adjustment. The logistic model smooths heuristic labels; it does not identify causal effects. README and reasoning output will use the honest name.

### F. Conformal Prediction (Day 5)

Nonconformity score: for each candidate in shortlist (not top 100), the residual between their composite score and the estimated threshold for their "tier" (based on career_feature buckets).

Conformal set: the bottom 80% of shortlist candidates serve as calibration set (pseudo-labeled by domain heuristics).

At inference:
- For each top-100 candidate, compute conformal p-value and CI width
- Map CI width → confidence label: narrow=high confidence, wide=marginal
- Use confidence label in reasoning template: "strong fit", "likely fit", "marginal fit"

### G. Template Reasoning

Format (1-2 sentences, no fabrication):

**Tier 1 (rank 1-10):** `{title} with {yoe:.1f} years, {N} years in applied ML/AI at product companies; shipped {system_type} systems; strong fit for the JD's production-deployment and ranking focus.`

**Tier 2 (rank 11-50):** `{title} at {company} with {yoe:.1f} years; {top_skill_1} and {top_skill_2} at {proficiency} level; good fit but {one_gap_or_concern}.`

**Tier 3 (rank 51-100):** `{title} with adjacent experience ({top_relevant_skill}); included as marginal fit — {concern}.`

Variables all pulled from actual profile fields. No hallucination is structurally possible.

---

## Regression Test (Day 1, before any other code)

`tests/test_no_keyword_stuffing.py`:
- Load `sample_submission.csv`
- Assert rank 1 candidate is NOT an "HR Manager" or "Marketing Manager" or "Content Writer" or "Operations Manager" etc.
- Assert no candidate with title ∈ NON_AI_TITLES appears in top 10 of any output produced by this system
- Parameterized so it can be run against the final `submission.csv` too

---

## File Structure

```
causal-graph-candidate-ranking/
├── PROJECT.md                          ← this file
├── ARCHITECTURE.md                     ← design (this file)
├── rank.py                             ← single entry point
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── India_runs_data_and_ai_challenge/   ← bundle (data files, not committed)
├── src/
│   ├── __init__.py
│   ├── parse.py                        ← feature extraction
│   ├── embed.py                        ← sentence-transformer embeddings
│   ├── index.py                        ← FAISS index build + query
│   ├── score.py                        ← composite scoring (MVP + full)
│   ├── ot_matching.py                  ← Sinkhorn OT (Days 3-4)
│   ├── causal.py                       ← disqualifier flags + debiasing (Days 3-4)
│   ├── conformal.py                    ← conformal calibration (Day 5)
│   ├── reasoning.py                    ← template reasoning generation
│   └── utils.py                        ← shared helpers
├── scripts/
│   └── precompute.py                   ← offline precompute (embeddings, index)
├── artifacts/                          ← .gitignored — precomputed artifacts
│   ├── candidate_embeddings.npy
│   ├── faiss_index.bin
│   ├── structured_features.parquet
│   ├── jd_embedding.npy
│   ├── jd_skill_matrix.npy
│   └── conformal_thresholds.pkl
├── tests/
│   ├── test_no_keyword_stuffing.py
│   └── test_output_format.py
└── .gitignore
```

---

## Compute Budget Estimation

| Step | Estimate | Notes |
|---|---|---|
| Load artifacts | ~10s | 147 MB embeddings + 80 MB features |
| FAISS query (k=5000) | <1s | Flat index at 100K |
| Feature scoring (5000 candidates) | ~2s | Vectorized pandas/numpy |
| OT Sinkhorn (5000 × 15 skills) | ~30-60s | Batched, POT library |
| Causal debiasing | ~1s | Logistic predict on 5000 |
| Sort + template reasoning | <1s | |
| **Total** | **~1-2 minutes** | Well within 5-minute budget |

---

## Known Risks

1. **OT cost at k=5000**: if Sinkhorn is too slow, reduce shortlist to k=2000 or reduce n_skills_per_candidate to 10.
2. **Plausibility score partial coverage**: structured signals (date-math + proficiency×duration) catch ~43-46 of ~80 expected honeypots. Remaining ~35 are undetectable from structured fields alone. They will naturally rank low due to irrelevant titles/careers — but this is probabilistic, not guaranteed. Do not rely solely on plausibility_score; the full scoring pipeline provides defense-in-depth.
3. **Conformal calibration without ground truth**: pseudo-labels may introduce bias — keep conformal CI as a secondary modifier, not the primary score driver.
4. **GNN (Day 6)**: if behind schedule, skip entirely — the OT + disqualifier-penalty system is already architecturally novel.
5. ~~**D1, D2, D4 text-pattern dependency**~~ **RESOLVED (Days 3-4 pre-work):** Empirical investigation over all 100K candidates found:
   - **D1**: text heuristic fires on 0 candidates; structured industry alternative also fires on 0 (no research/academia industry exists in dataset). Dropped — non-applicable.
   - **D2**: text heuristic fires on 0 candidates (only 64 mention LLM keywords; none without pre-LLM hits). Dropped — non-applicable.
   - **D4**: was already purely structural in `parse.py` (duration_months + n_roles, no free text used). Renamed `frequent_job_hopper` to accurately describe what it measures (frequent switching, not title elevation). 1,382 flagged; 9 are ML/AI titles (minor false-positive risk, documented).
   - Net result: disqualifier-penalty scoring will use D3, D4(renamed), D5, D6, D7, D8 only. No text-pattern dependency in any active flag.

---

## Approval Checklist (fill before writing code)

- [ ] Architecture reviewed and approved
- [ ] No LLM calls at inference — confirmed by design
- [ ] Regression test written before scoring code
- [ ] Each phase boundary produces a valid submission
- [ ] OT implementation benchmarked on sample before full precompute
