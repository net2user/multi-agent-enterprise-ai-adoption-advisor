"""
Cache Generator

Run this once, locally, to generate genuine cached results for all
eight synthetic use cases plus the portfolio ranking, saved to
data/cached_assessments.json. app.py reads this file directly for any
preset use case or the default Portfolio View, no live API calls
needed for those paths going forward. "Write my own use case" and the
optional "Run live portfolio assessment anyway" button in Portfolio
View still call the API live, on purpose.

Saves incrementally after each use case, and skips any use case
already present in the existing cache file, so a rate limit failure
partway through doesn't lose completed work or require a full re-run.

Set MAX_NEW_USE_CASES to deliberately cap how many NEW (not already
cached) use cases run in this invocation, leaving remaining token
budget free for other work the same day. Defaults to no cap.
Example: MAX_NEW_USE_CASES=5 PYTHONPATH=src python generate_cache.py

Cost note: with RAG grounding now active, real observed cost is closer
to 13,000-14,000 tokens per use case (seven calls, several of them now
carrying retrieved regulatory context), not the original ~6,100 token
estimate from before grounding was added. Groq's llama-3.3-70b-versatile
daily limit is 100,000 tokens, so a full eight use case run (~110K)
does not fit in one day even fresh. Check your Groq usage page first.
"""

import json
import os
import time

from orchestrator import run_single_use_case_assessment
from executive_summary_agent import generate_executive_summary
from implementation_roadmap_agent import generate_implementation_roadmap
from portfolio_agent import prioritize_portfolio

CACHE_FILE = "data/cached_assessments.json"
MAX_NEW_USE_CASES = int(os.environ.get("MAX_NEW_USE_CASES", "999"))

import os as _os
import time as _time
import requests as _requests

_rag_url = _os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
print(f"Warming up RAG service at {_rag_url}...")
for _attempt in range(10):
    try:
        _r = _requests.get(f"{_rag_url}/health", timeout=15)
        if _r.status_code == 200:
            print("RAG service is awake and ready.")
            break
    except Exception:
        pass
    print(f"  Not ready yet, waiting (attempt {_attempt + 1}/10)...")
    _time.sleep(5)
else:
    print("WARNING: RAG service did not respond after 50s. Proceeding anyway, agents will fall back to ungrounded if this continues.")

with open("data/use_case_portfolio.json") as f:
    portfolio = json.load(f)["use_cases"]

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        existing = json.load(f)
    single_use_case_cache = existing.get("single_use_case", {})
    print(f"Found existing cache with {len(single_use_case_cache)} use case(s), will skip those and resume.")
else:
    single_use_case_cache = {}

per_use_case_assessments = {}
scored_for_ranking = []
new_this_run = 0

for uc in portfolio:
    if uc["id"] in single_use_case_cache:
        cached = single_use_case_cache[uc["id"]]
        per_use_case_assessments[uc["id"]] = cached["assessment"]
        scored_for_ranking.append({
            "use_case_id": uc["id"],
            "title": uc["title"],
            "value_score": cached["assessment"]["value"]["value_score"],
            "risk_score": cached["assessment"]["risk"]["risk_score"],
            "complexity_score": cached["assessment"]["architecture"]["complexity_score"],
            "adoption_score": cached["assessment"]["adoption"]["adoption_score"],
        })
        continue

    if new_this_run >= MAX_NEW_USE_CASES:
        print(f"Reached MAX_NEW_USE_CASES cap ({MAX_NEW_USE_CASES}), stopping here to leave token budget for other work today.")
        break

    print(f"Running {uc['id']}: {uc['title']}...")

    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "estimated_annual_cost_usd": uc.get("estimated_annual_cost_usd"),
        "current_process_maturity": uc.get("current_process_maturity"),
        "data_sensitivity": uc.get("data_sensitivity"),
        "regulatory_exposure": uc.get("regulatory_exposure"),
        "integration_points": uc.get("integration_points"),
        "vendor": uc.get("vendor"),
        "stakeholders": uc.get("stakeholders"),
    }

    assessment = run_single_use_case_assessment(uc["description"], context)

    summary = generate_executive_summary(
        uc["description"],
        assessment["value"],
        assessment["risk"],
        assessment["architecture"],
        assessment["adoption"],
        assessment.get("data_readiness"),
    )

    roadmap = generate_implementation_roadmap(
        uc["description"],
        assessment["value"],
        assessment["risk"],
        assessment["architecture"],
        assessment["adoption"],
        assessment.get("data_readiness"),
    )

    single_use_case_cache[uc["id"]] = {
        "assessment": assessment,
        "summary": summary,
        "roadmap": roadmap,
    }
    per_use_case_assessments[uc["id"]] = assessment

    scored_for_ranking.append({
        "use_case_id": uc["id"],
        "title": uc["title"],
        "value_score": assessment["value"]["value_score"],
        "risk_score": assessment["risk"]["risk_score"],
        "complexity_score": assessment["architecture"]["complexity_score"],
        "adoption_score": assessment["adoption"]["adoption_score"],
    })
    new_this_run += 1

    with open(CACHE_FILE, "w") as f:
        json.dump({
            "single_use_case": single_use_case_cache,
            "portfolio_view": {
                "ranked_portfolio": None,
                "per_use_case_assessments": per_use_case_assessments,
            },
        }, f, indent=2)

    print(f"  Done, saved to {CACHE_FILE}. Pausing briefly before the next use case...")
    time.sleep(3)

if len(scored_for_ranking) == len(portfolio):
    print("All use cases present. Running portfolio ranking...")
    ranked_portfolio = prioritize_portfolio(scored_for_ranking)

    with open(CACHE_FILE, "w") as f:
        json.dump({
            "single_use_case": single_use_case_cache,
            "portfolio_view": {
                "ranked_portfolio": ranked_portfolio,
                "per_use_case_assessments": per_use_case_assessments,
            },
        }, f, indent=2)

    print(f"\nDone. Saved to {CACHE_FILE}")
    print(f"Cached {len(single_use_case_cache)} use cases plus one portfolio ranking.")
else:
    print(f"\n{len(scored_for_ranking)}/{len(portfolio)} use cases cached so far ({new_this_run} new this run).")
    print("Portfolio ranking skipped until all use cases are cached. Re-run this script (same command, or with a fresh MAX_NEW_USE_CASES) once ready, it will resume from here.")
