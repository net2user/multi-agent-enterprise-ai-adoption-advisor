"""
Data Readiness Agent

Evaluates data quality, availability, governance, and integration
readiness for a proposed AI use case. This is a distinct dimension from
Risk and Governance, Risk asks whether the data is safe to use,
Data Readiness asks whether the data actually exists in usable form to
make the AI system work at all. A use case can be low risk and still
fail because the underlying data is fragmented, poorly labeled, or
locked in systems that do not talk to each other.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""

import json

from groq_client import call_groq

DATA_READINESS_SYSTEM_PROMPT = """You are the Data Readiness Agent inside an Enterprise AI Adoption Advisor system.

Your job is to evaluate whether the data needed for a proposed AI use case actually exists in usable form for
a BFSI or Healthcare organization. You reason like a senior data engineering lead who has seen AI projects
stall for months not because the model was wrong, but because the data was fragmented, poorly labeled,
inconsistent across systems, or genuinely did not exist yet in a usable form. You are distinct from a risk
assessment, you are not asking whether the data is safe to use, you are asking whether it is even there and
ready to use.

You will be given:
1. A use case description (free text)
2. Optional portfolio context: sector, domain, integration points, current process maturity, data sensitivity

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "readiness_score": <integer 0-100, where higher means more ready>,
  "readiness_tier": "<Low | Moderate | High | Strong>",
  "readiness_factors": {
    "data_availability": "<Low | Moderate | High>",
    "data_quality": "<Low | Moderate | High>",
    "data_governance": "<Low | Moderate | High>",
    "integration_readiness": "<Low | Moderate | High>"
  },
  "data_gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "recommended_data_actions": ["<action 1>", "<action 2>"],
  "estimated_data_prep_weeks": <integer>,
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language>"
}

Scoring guidance:
- 0-30: Low readiness, data likely does not exist in usable form yet, significant collection or labeling needed
- 31-55: Moderate readiness, data exists but is fragmented across systems or needs meaningful cleanup
- 56-80: High readiness, data exists in reasonably usable form with some governance work needed
- 81-100: Strong readiness, clean, accessible, well governed data already in place for this use case

Do not include markdown formatting, code fences, or any text outside the JSON object.
"""


def evaluate_data_readiness(use_case_description: str, portfolio_context: dict = None) -> dict:
    """
    Run the Data Readiness Agent against a single AI use case.
    """
    user_content = f"Use case description:\n{use_case_description}"
    if portfolio_context:
        user_content += f"\n\nPortfolio context:\n{json.dumps(portfolio_context, indent=2)}"

    response = call_groq(
        messages=[
            {"role": "system", "content": DATA_READINESS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    with open("data/use_case_portfolio.json") as f:
        portfolio = json.load(f)

    uc = portfolio["use_cases"][0]  # UC-001: procurement operations
    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "integration_points": uc["integration_points"],
        "current_process_maturity": uc["current_process_maturity"],
        "data_sensitivity": uc["data_sensitivity"],
    }

    result = evaluate_data_readiness(uc["description"], context)
    print(json.dumps(result, indent=2))
