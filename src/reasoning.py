"""
Template-based reasoning generation.
All values pulled from real profile fields — no LLM, no fabrication possible.
"""

from __future__ import annotations

import json
from datetime import date

TODAY = date(2026, 6, 28)

INDIA_PREFERRED = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
                   "gurugram", "gurgaon", "delhi ncr"}


def _top_relevant_skills(skills_json: str, n: int = 2) -> list[str]:
    """Return top-n skill names from the pre-ranked skills_json."""
    try:
        skills = json.loads(skills_json)
        return [s["name"] for s in skills[:n] if s.get("name")]
    except (json.JSONDecodeError, TypeError):
        return []


def _days_since_active(last_active_date: str) -> int:
    if not last_active_date:
        return 999
    try:
        d = date.fromisoformat(last_active_date[:10])
        return (TODAY - d).days
    except ValueError:
        return 999


def _location_phrase(row) -> str:
    loc = str(row.get("location", "")).strip()
    country = str(row.get("country", "")).strip()
    relocate = row.get("willing_to_relocate", False)
    if loc.lower() in INDIA_PREFERRED:
        return f"based in {loc}"
    if country == "India" and relocate:
        return f"India-based ({loc}), open to relocation"
    if country == "India":
        return f"India-based ({loc})"
    if relocate:
        return f"based in {loc}, willing to relocate"
    return f"based in {loc}"


def _availability_phrase(row) -> str | None:
    days = _days_since_active(str(row.get("last_active_date", "")))
    rrr = float(row.get("recruiter_response_rate", 0))
    notice = int(row.get("notice_period_days", 90))

    if not row.get("open_to_work", False) and days > 90:
        return "not actively looking and inactive >90 days"
    if days > 120:
        return f"inactive {days} days"
    if notice <= 30:
        return "available quickly (≤30d notice)"
    if notice >= 90:
        return f"{notice}d notice"
    return None


def _concern_phrase(row) -> str | None:
    """Surface one honest concern if present."""
    if row.get("d6_pure_services", False):
        return "entire career at IT services — no product company experience on record"
    if row.get("d7_cv_speech", False):
        return "primary background in CV/speech — NLP/IR depth unclear"
    if row.get("d3_no_prod_code", False):
        return "current role appears architecture/leadership-focused, not hands-on coding"
    if row.get("d4_title_chaser", False):
        return "multiple short stints with rapid title progression"
    days = _days_since_active(str(row.get("last_active_date", "")))
    if days > 180:
        return f"platform inactive {days} days — availability uncertain"
    return None


def generate_reasoning(row, rank: int) -> str:
    """
    1-2 sentence reasoning from real fields only.
    Tier assigned by rank bucket.
    """
    title = str(row.get("current_title", "")).strip()
    yoe = float(row.get("yoe", 0))
    company = str(row.get("current_company", "")).strip()
    top_skills = _top_relevant_skills(str(row.get("skills_json", "[]")))
    skills_str = " and ".join(top_skills) if top_skills else "relevant technical skills"
    loc_phrase = _location_phrase(row)
    avail_phrase = _availability_phrase(row)
    concern = _concern_phrase(row)
    prod_months = int(row.get("product_months", 0))
    ml_months = int(row.get("ml_ai_months", 0))

    if rank <= 10:
        # Tier 1: specific, positive, with one concrete detail
        detail = ""
        if ml_months >= 36:
            detail = f"{ml_months // 12}+ years in applied ML/AI roles"
        elif prod_months >= 36:
            detail = f"{prod_months // 12}+ years at product companies"
        else:
            detail = f"{skills_str} at {company}" if company else skills_str

        sentence1 = (
            f"{title} with {yoe:.1f} years of experience; {detail}; "
            f"{loc_phrase}."
        )
        if avail_phrase:
            sentence2 = f"Platform signal: {avail_phrase}."
        elif concern:
            sentence2 = f"Note: {concern}."
        else:
            sentence2 = f"Strong fit for the JD's production-deployment and ranking focus."
        return f"{sentence1} {sentence2}"

    elif rank <= 50:
        # Tier 2: balanced, note one gap if present
        sentence1 = (
            f"{title} at {company} with {yoe:.1f} years; "
            f"{skills_str} ({str(row.get('top_skill_1_prof', '')).replace('_', ' ')}-level); "
            f"{loc_phrase}."
        )
        if concern:
            sentence2 = f"Concern: {concern}."
        elif avail_phrase:
            sentence2 = f"Availability: {avail_phrase}."
        else:
            sentence2 = f"Solid profile, ranked below top candidates on ML/AI role depth."
        return f"{sentence1} {sentence2}"

    else:
        # Tier 3: honest about marginal fit
        sentence1 = (
            f"{title} with {yoe:.1f} years; "
            f"top skills include {skills_str}; {loc_phrase}."
        )
        if concern:
            sentence2 = f"Below cutoff: {concern}."
        else:
            sentence2 = "Adjacent experience — ranked as marginal fit given limited ML/AI role history."
        return f"{sentence1} {sentence2}"


def add_reasoning_column(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df = df.copy()
    df["reasoning"] = [
        generate_reasoning(row, int(row["rank"]))
        for _, row in df.iterrows()
    ]
    return df
