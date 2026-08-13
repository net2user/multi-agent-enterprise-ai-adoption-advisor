"""
Implementation Roadmap Agent

Takes the completed Value, Risk, Architecture, Adoption, and Data
Readiness assessments for a use case and generates a phased
implementation roadmap, 30, 60, 90 days, then 6 months, then 1 year.
This agent does not re-score anything, it reads the timelines the
other agents already estimated, the Architecture agent's pilot weeks
and the Data Readiness agent's data prep weeks specifically, and
sequences them into a coherent plan rather than inventing new numbers
from scratch.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""

import json

from groq_client import call_groq

IMPLEMENTATION_ROADMAP_SYSTEM_PROMPT = """You are the Implementation Roadmap Agent inside an Enterprise AI Adoption Advisor system.

Your job is to turn a completed AI use case assessment into a phased implementation roadmap for a BFSI or
Healthcare organization. You reason like a senior program manager who has actually delivered AI pilots into
production, not like someone padding a generic template. You read the timelines the other agents already
estimated, the Architecture agent's estimated weeks to pilot and the Data Readiness agent's estimated data
preparation weeks specifically, and sequence real work around them, data prep and governance work generally
needs to happen before or alongside integration work, not after.

You will be given the use case description and the completed Value, Risk, Architecture, Adoption, and Data
Readiness agent assessments as JSON.

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "roadmap_phases": [
    {
      "timeframe": "Days 1-30",
      "focus": "<short phrase, what this phase is actually about>",
      "key_milestones": ["<milestone 1>", "<milestone 2>"],
      "dependencies": ["<dependency 1>"]
    },
    {
      "timeframe": "Days 31-60",
      "focus": "<short phrase>",
      "key_milestones": ["<milestone 1>", "<milestone 2>"],
      "dependencies": ["<dependency 1>"]
    },
    {
      "timeframe": "Days 61-90",
      "focus": "<short phrase>",
      "key_milestones": ["<milestone 1>", "<milestone 2>"],
      "dependencies": ["<dependency 1>"]
    },
    {
      "timeframe": "Months 4-6",
      "focus": "<short phrase>",
      "key_milestones": ["<milestone 1>", "<milestone 2>"],
      "dependencies": ["<dependency 1>"]
    },
    {
      "timeframe": "Months 7-12",
      "focus": "<short phrase>",
      "key_milestones": ["<milestone 1>", "<milestone 2>"],
      "dependencies": ["<dependency 1>"]
    }
  ],
  "critical_path_dependencies": ["<the one or two things that would delay everything else if they slip>"],
  "estimated_total_timeline_months": <integer, from kickoff to full production adoption>,
  "success_metrics": ["<metric 1>", "<metric 2>"],
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language, referencing the pilot and data prep timelines this roadmap is built around>"
}

Do not include markdown formatting, code fences, or any text outside the JSON object.
"""


def generate_implementation_roadmap(use_case_description: str, value_result: dict, risk_result: dict,
                                     architecture_result: dict, adoption_result: dict,
                                     data_readiness_result: dict = None) -> dict:
    """
    Run the Implementation Roadmap Agent against the completed agent assessments
    for a single use case. data_readiness_result is optional so this still works
    if called before Data Readiness results are available.
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
            {"role": "system", "content": IMPLEMENTATION_ROADMAP_SYSTEM_PROMPT},
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

    roadmap = generate_implementation_roadmap(
        uc["description"],
        assessment["value"],
        assessment["risk"],
        assessment["architecture"],
        assessment["adoption"],
        assessment.get("data_readiness"),
    )

    print(json.dumps(roadmap, indent=2))
