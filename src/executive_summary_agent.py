"""
Executive Summary Agent

Reads the outputs of the Value, Risk, Architecture, Adoption, and Data
Readiness agents for a single use case and synthesizes them into a
board level briefing. This agent does not re-score anything, its job
is judgment and framing, turning five separate assessments into one
coherent recommendation the way a senior advisor would walk a board
through it in a single slide.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""

import json

from groq_client import call_groq

EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """You are the Executive Summary Agent inside an Enterprise AI Adoption Advisor system.

Your job is to synthesize the outputs of five specialist agents, Value, Risk and Governance, Architecture,
Change Management and Adoption, and Data Readiness, into a single board level briefing for a BFSI or
Healthcare organization. You reason like a senior advisor walking a board or executive committee through a
recommendation in one slide, not like someone restating five separate reports back to back. You make a
clear call, fund it, fund it with conditions, delay it, or do not fund it, and you say why in plain language.
Data Readiness is distinct from Risk, it is not about whether the data is safe, it is about whether usable
data actually exists yet to make the system work, weigh it accordingly rather than folding it into risk.

You will be given the use case description and the five completed agent assessments as JSON.

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "overall_recommendation": "<Fund | Fund with Conditions | Delay | Do Not Fund>",
  "executive_headline": "<one sentence a board member could repeat from memory>",
  "briefing_paragraph": "<4-6 sentence board level briefing in plain, consulting-grade language, weaving together value, risk, feasibility, data readiness, and adoption readiness into one coherent narrative, not a list>",
  "top_conditions_for_funding": ["<condition 1>", "<condition 2>"],
  "key_tension": "<the single most important tradeoff a board member should understand, in one sentence>",
  "confidence": "<Low | Medium | High>"
}

Do not include markdown formatting, code fences, or any text outside the JSON object.
"""


def generate_executive_summary(use_case_description: str, value_result: dict, risk_result: dict,
                                architecture_result: dict, adoption_result: dict,
                                data_readiness_result: dict = None) -> dict:
    """
    Run the Executive Summary Agent against the completed agent assessments
    for a single use case. data_readiness_result is optional so existing
    calls without it still work without breaking.
    """
    user_content = (
        f"Use case description:\n{use_case_description}\n\n"
        f"Value agent assessment:\n{json.dumps(value_result, indent=2)}\n\n"
        f"Risk and Governance agent assessment:\n{json.dumps(risk_result, indent=2)}\n\n"
        f"Architecture agent assessment:\n{json.dumps(architecture_result, indent=2)}\n\n"
        f"Adoption agent assessment:\n{json.dumps(adoption_result, indent=2)}"
    )

    if data_readiness_result:
        user_content += f"\n\nData Readiness agent assessment:\n{json.dumps(data_readiness_result, indent=2)}"

    response = call_groq(
        messages=[
            {"role": "system", "content": EXECUTIVE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    from orchestrator import run_single_use_case_assessment

    with open("data/use_case_portfolio.json") as f:
        portfolio = json.load(f)

    uc = portfolio["use_cases"][0]  # UC-001: procurement operations
    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "estimated_annual_cost_usd": uc["estimated_annual_cost_usd"],
        "current_process_maturity": uc["current_process_maturity"],
        "data_sensitivity": uc["data_sensitivity"],
        "regulatory_exposure": uc["regulatory_exposure"],
        "integration_points": uc["integration_points"],
        "vendor": uc["vendor"],
        "stakeholders": uc["stakeholders"],
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

    print(json.dumps(summary, indent=2))
