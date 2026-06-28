"""
Parse candidates.jsonl into a structured feature DataFrame.

All features derived from structured fields only — no regex over free-text descriptions
except where explicitly documented as heuristic (D2, D5, D8 flags, marked with HEURISTIC).

Output: artifacts/structured_features.parquet
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

TODAY = date(2026, 6, 28)

# IT-services companies whose entire career = disqualifier D6
IT_SERVICES_COMPANIES = {
    "tcs", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "niit technologies",
    "mastech", "igate", "patni", "syntel",
}

# CV/speech/robotics industry/title keywords — D7 signal (structured titles only)
CV_SPEECH_ROBOTICS_TITLES = {
    "computer vision", "speech recognition", "robotics", "autonomous",
    "image processing", "object detection", "slam", "lidar",
}

# ML/AI role titles for estimating applied_ml_years
ML_AI_TITLES = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research scientist", "applied scientist",
    "deep learning", "llm", "ai researcher", "ml researcher",
}

PROFICIENCY_VALUE = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# HEURISTIC: text signals for D2 (LLM-only AI) — approximate, documented in README
LANGCHAIN_KEYWORDS = re.compile(
    r"\b(langchain|llamaindex|llama[_-]index|langsmith|openai api|gpt-3|gpt-4|chatgpt"
    r"|prompt engineer|rag pipeline|vector store)\b",
    re.IGNORECASE,
)
PRE_LLM_ML_KEYWORDS = re.compile(
    r"\b(xgboost|lightgbm|random forest|svm|sklearn|scikit.learn|gradient boost"
    r"|recommendation system|ranking|search|retrieval|embedding|bert|transformer"
    r"|pytorch|tensorflow|keras|fine.tun|feature engineer)\b",
    re.IGNORECASE,
)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _company_is_services(name: str) -> bool:
    return name.strip().lower() in IT_SERVICES_COMPANIES


def _is_product_company(role: dict) -> bool:
    """A role is at a product company if industry is NOT IT Services and company is not services."""
    if _company_is_services(role.get("company", "")):
        return False
    it_industries = {"it services", "information technology", "outsourcing", "bpo", "consulting"}
    industry = role.get("industry", "").strip().lower()
    return industry not in it_industries


def _proficiency_inconsistency(skills: list[dict]) -> float:
    """
    Honeypot signal: max gap between declared proficiency and duration_months.
    expert needs ~24mo; advanced ~12mo; intermediate ~3mo.
    Returns worst normalized gap across all skills (0 = consistent, 1 = maximally inconsistent).
    """
    EXPECTED_MONTHS = {"beginner": 0, "intermediate": 3, "advanced": 12, "expert": 24}
    MAX_INCONSISTENCY = 24  # expert at 0 months = gap of 24
    worst = 0.0
    for s in skills:
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        expected = EXPECTED_MONTHS[prof]
        gap = max(0, expected - dur)
        worst = max(worst, gap / MAX_INCONSISTENCY)
    return worst


def _date_math_inconsistency(career: list[dict], declared_yoe: float) -> float:
    """
    Honeypot signal: sum of duration_months vs declared years_of_experience.
    Returns normalized excess (0 = consistent, 1 = maximally inconsistent).
    """
    sum_months = sum(r.get("duration_months", 0) for r in career)
    declared_months = declared_yoe * 12
    excess = max(0, sum_months - declared_months - 12)  # 12mo tolerance for overlaps
    return min(1.0, excess / 120)  # cap at 10yr excess = 1.0


def _plausibility_score(career: list[dict], skills: list[dict], declared_yoe: float) -> float:
    """
    Combined honeypot plausibility score in [0, 1].
    1 = fully plausible, 0 = clear honeypot.
    """
    date_penalty = _date_math_inconsistency(career, declared_yoe)
    prof_penalty = _proficiency_inconsistency(skills)
    # Weight: date math is a stronger signal (harder to explain away)
    combined_penalty = 0.6 * date_penalty + 0.4 * prof_penalty
    return round(1.0 - combined_penalty, 4)


def _availability_score(sig: dict) -> float:
    """
    Availability modifier in [0, 1] from redrob_signals.
    Applied as score^0.3 at ranking time, so this doesn't dominate.
    """
    score = 0.0

    # Open to work: most direct signal
    if sig.get("open_to_work_flag", False):
        score += 0.30

    # Recency: exponential decay, half-life ~90 days
    last_active = _parse_date(sig.get("last_active_date", ""))
    if last_active:
        days_since = (TODAY - last_active).days
        score += 0.25 * max(0.0, 1.0 - days_since / 180)

    # Recruiter response rate
    rrr = float(sig.get("recruiter_response_rate", 0))
    score += 0.20 * rrr

    # Notice period: 0 days = 1.0, 90 days = 0.5, 180 days = 0.0
    notice = int(sig.get("notice_period_days", 90))
    score += 0.15 * max(0.0, 1.0 - notice / 180)

    # Interview completion
    icr = float(sig.get("interview_completion_rate", 0))
    score += 0.10 * icr

    return round(min(1.0, score), 4)


def _location_score(profile: dict, sig: dict) -> float:
    """Score location fit for Noida/Pune/Hyderabad/Mumbai/Delhi NCR."""
    PREFERRED = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
                 "delhi ncr", "gurugram", "gurgaon", "faridabad", "noida extension"}
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    if any(city in loc for city in PREFERRED):
        return 1.0
    if country == "india" and sig.get("willing_to_relocate", False):
        return 0.7
    if country == "india":
        return 0.4
    if sig.get("willing_to_relocate", False):
        return 0.2
    return 0.0


def _career_features(career: list[dict], profile: dict) -> dict:
    """Extract structured career signals."""
    product_months = 0
    services_only = True
    has_current_product = False
    all_companies = []
    title_history = []  # sorted by start_date for D4

    ml_ai_months = 0
    current_role_months = 0
    current_title = profile.get("current_title", "").lower()

    for role in sorted(career, key=lambda r: r.get("start_date", "0000")):
        company = role.get("company", "")
        all_companies.append(company.strip().lower())
        title = role.get("title", "").lower()
        dur = role.get("duration_months", 0)
        is_current = role.get("is_current", False)

        if _is_product_company(role):
            product_months += dur
            services_only = False
            if is_current:
                has_current_product = True
        elif is_current:
            pass  # currently at services, but past product is OK

        # ML/AI role detection by title (structured field)
        if any(kw in title for kw in ML_AI_TITLES):
            ml_ai_months += dur

        if is_current:
            current_role_months = dur

        title_history.append((role.get("start_date", ""), title, dur))

    # D6: pure services — all companies are IT services AND no product company ever
    d6_pure_services = services_only and all(
        _company_is_services(c) for c in all_companies
    )

    # D3: no prod code in last 18mo — current title suggests arch/lead/principal
    ARCH_TITLES = {"architect", "tech lead", "engineering manager", "vp of", "head of", "director"}
    d3_no_prod_code = (
        current_role_months >= 18
        and any(t in current_title for t in ARCH_TITLES)
    )

    # D4: title chaser — detect rapid promotions via title hops in short stints
    # Structured: look for Short (<18mo) stints with seniority-jump titles
    SENIORITY_LEVELS = ["junior", "associate", "mid", "senior", "staff", "principal", "distinguished"]
    d4_title_chaser = False
    if len(title_history) >= 3:
        short_jumps = 0
        for _, _, dur in title_history:
            if 0 < dur < 18:
                short_jumps += 1
        # If more than half of roles are very short stints → title-chasing pattern
        d4_title_chaser = short_jumps >= len(title_history) // 2 + 1 and len(title_history) >= 4

    # D7: CV/speech/robotics background — structured: current_title and industry
    d7_cv_speech = any(kw in current_title for kw in CV_SPEECH_ROBOTICS_TITLES)

    return {
        "product_months": product_months,
        "services_only": services_only,
        "has_current_product": has_current_product,
        "ml_ai_months": ml_ai_months,
        "current_role_months": current_role_months,
        "d3_no_prod_code": d3_no_prod_code,
        "d4_title_chaser": d4_title_chaser,
        "d6_pure_services": d6_pure_services,
        "d7_cv_speech": d7_cv_speech,
    }


def _skill_features(skills: list[dict]) -> dict:
    """Top-weighted skills (proficiency × log(1+duration)) for OT matching later."""
    weighted = []
    for s in skills:
        pv = PROFICIENCY_VALUE.get(s.get("proficiency", "beginner"), 1)
        dur = s.get("duration_months", 0)
        import math
        w = pv * math.log1p(dur)
        weighted.append((s["name"], w, s.get("proficiency", "beginner"), dur))
    weighted.sort(key=lambda x: -x[1])
    top3 = weighted[:3]
    return {
        "top_skill_1": top3[0][0] if len(top3) > 0 else "",
        "top_skill_1_prof": top3[0][2] if len(top3) > 0 else "",
        "top_skill_2": top3[1][0] if len(top3) > 1 else "",
        "top_skill_3": top3[2][0] if len(top3) > 2 else "",
        "n_skills": len(skills),
        "n_advanced_expert": sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert")),
        "skills_json": json.dumps([
            {"name": name, "proficiency": prof, "duration_months": dur}
            for name, _w, prof, dur in weighted[:20]
        ]),
    }


def _heuristic_flags(career: list[dict], sig: dict) -> dict:
    """
    HEURISTIC text-pattern flags for D1, D2, D5, D8.
    These depend on career_history.description free text.
    Documented as approximate in README; not used in MVP scoring.
    Only activated for Days 3-4 disqualifier-penalty scoring.
    """
    all_desc = " ".join(r.get("description", "") for r in career)

    # D1: no production role — heuristic: descriptions mention only research/academic terms
    research_terms = re.compile(
        r"\b(research|paper|arxiv|publication|lab|phd|academic|thesis|dissertation)\b",
        re.IGNORECASE,
    )
    prod_terms = re.compile(
        r"\b(deployed|production|shipped|users|scale|latency|infra|api|service|pipeline"
        r"|platform|a/b test|rollout|real.time|million|billion)\b",
        re.IGNORECASE,
    )
    research_hits = len(research_terms.findall(all_desc))
    prod_hits = len(prod_terms.findall(all_desc))
    d1_pure_research = research_hits > 3 and prod_hits < 2

    # D2: LLM-only AI (HEURISTIC)
    llm_hits = len(LANGCHAIN_KEYWORDS.findall(all_desc))
    pre_llm_hits = len(PRE_LLM_ML_KEYWORDS.findall(all_desc))
    d2_llm_only = llm_hits >= 3 and pre_llm_hits < 2

    # D5: framework enthusiast — low github + high LLM tutorial pattern
    github = float(sig.get("github_activity_score", -1))
    github_effective = github if github >= 0 else 0.0
    d5_framework_enthusiast = github_effective < 20 and llm_hits >= 2

    # D8: closed-source only — no github AND large-only companies
    d8_closed_source = (
        github < 0
        and sig.get("linkedin_connected", False) is False
        and not sig.get("verified_email", False)
    )

    return {
        "d1_pure_research_heuristic": d1_pure_research,
        "d2_llm_only_heuristic": d2_llm_only,
        "d5_framework_enthusiast_heuristic": d5_framework_enthusiast,
        "d8_closed_source_heuristic": d8_closed_source,
    }


def parse_candidate(c: dict) -> dict:
    profile = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})
    edu = c.get("education", [])

    yoe = float(profile.get("years_of_experience", 0))
    career_feats = _career_features(career, profile)
    skill_feats = _skill_features(skills)
    heuristic = _heuristic_flags(career, sig)

    # Redrob signals
    github = float(sig.get("github_activity_score", -1))
    oar = float(sig.get("offer_acceptance_rate", -1))

    # Education tier
    edu_tiers = [e.get("tier", "unknown") for e in edu]
    best_tier = (
        "tier_1" if "tier_1" in edu_tiers else
        "tier_2" if "tier_2" in edu_tiers else
        "tier_3" if "tier_3" in edu_tiers else
        "tier_4" if "tier_4" in edu_tiers else "unknown"
    )

    row = {
        "candidate_id": c["candidate_id"],
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "yoe": yoe,
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        # Plausibility / honeypot
        "plausibility_score": _plausibility_score(career, skills, yoe),
        # Location / availability
        "location_score": _location_score(profile, sig),
        "availability_score": _availability_score(sig),
        # Career
        **career_feats,
        # Skills
        **skill_feats,
        # Redrob signals (raw, for reasoning templates)
        "open_to_work": sig.get("open_to_work_flag", False),
        "recruiter_response_rate": float(sig.get("recruiter_response_rate", 0)),
        "notice_period_days": int(sig.get("notice_period_days", 90)),
        "github_activity_score": github,
        "github_effective": max(0.0, github),
        "profile_completeness": float(sig.get("profile_completeness_score", 0)),
        "last_active_date": sig.get("last_active_date", ""),
        "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "offer_acceptance_rate": oar,
        "interview_completion_rate": float(sig.get("interview_completion_rate", 0)),
        "salary_min_lpa": float(sig.get("expected_salary_range_inr_lpa", {}).get("min", 0)),
        "salary_max_lpa": float(sig.get("expected_salary_range_inr_lpa", {}).get("max", 0)),
        # Education
        "best_edu_tier": best_tier,
        # Heuristic flags (Days 3-4 only, not used in MVP)
        **heuristic,
        # Raw text for embedding (precompute step uses this)
        "embed_text": (
            f"{profile.get('headline', '')} | "
            f"{profile.get('summary', '')} | "
            + " | ".join(r.get("description", "") for r in career[:3])
        ),
    }
    return row


def build_feature_table(candidates_path: Path, out_path: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    with open(candidates_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            rows.append(parse_candidate(c))
            if verbose and (i + 1) % 10000 == 0:
                print(f"  Parsed {i + 1:,} candidates...")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False, compression="snappy")
    if verbose:
        print(f"Feature table saved: {out_path}  ({len(df):,} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    import sys
    candidates = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("India_runs_data_and_ai_challenge/candidates.jsonl")
    out = Path("artifacts/structured_features.parquet")
    out.parent.mkdir(exist_ok=True)
    build_feature_table(candidates, out)
