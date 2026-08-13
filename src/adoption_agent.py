"""
Change Management and Adoption Agent
Evaluates organizational readiness and adoption risk for a proposed AI
use case, distinct from technical risk. This is where AI initiatives most
often fail in practice, not because the model was wrong, but because the
organization was not ready to act on what it produced.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""
import json

from groq_client import call_groq

ADOPTION_AGENT_SYSTEM_PROMPT = """You are the Change Management and Adoption Agent inside an Enterprise AI Adoption Advisor system.
Your job is to evaluate organizational readiness for a proposed AI use case in BFSI or Healthcare. You reason
like a senior change management advisor who has watched technically sound AI projects fail because the
organization was not ready, not because the model was wrong. You focus on people, process, and incentive
alignment, not technology.
You will be given:
1. A use case description (free text)
2. Optional portfolio context: sector, domain, stakeholders, current process maturity
Return your evaluation as strict JSON with this exact schema, and nothing else:
{
  "adoption_score": <integer 0-100, where higher means more ready for adoption>,
  "adoption_tier": "<Low | Moderate | High | Strong>",
  "readiness_factors": {
    "leadership_sponsorship": "<Low | Moderate | High>",
    "process_disruption": "<Low | Moderate | High>",
    "workforce_impact": "<Low | Moderate | High>",
    "incentive_alignment": "<Low | Moderate | High>"
  },
  "adoption_barriers": ["<barrier 1>", "<barrier 2>", "<barrier 3>"],
  "change_management_actions": ["<action 1>", "<action 2>"],
  "stakeholder_groups_affected": ["<group 1>", "<group 2>"],
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language>"
}
Scoring guidance:
- 0-30: Low readiness, significant workforce disruption, unclear ownership, no sponsorship signal
- 31-55: Moderate readiness, some process change, needs active change management to succeed
- 56-80: High readiness, clear stakeholder buy-in likely, moderate process change
- 81-100: Strong readiness, minimal disruption, clear ownership, natural fit with current ways of working
Do not include markdown formatting, code fences, or any text outside the JSON object.
"""

def evaluate_adoption(use_case_description: str, portfolio_context: dict = None) -> dict:
    """
    Run the Change Management and Adoption Agent against a single AI use case.
    """
    user_content = f"Use case description:\n{use_case_description}"
    if portfolio_context:
        user_content += f"\n\nPortfolio context:\n{json.dumps(portfolio_context, indent=2)}"
    response = call_groq(
        messages=[
            {"role": "system", "content": ADOPTION_AGENT_SYSTEM_PROMPT},
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
        "stakeholders": uc["stakeholders"],
        "current_process_maturity": uc["current_process_maturity"],
    }
    result = evaluate_adoption(uc["description"], context)
    print(json.dumps(result, indent=2))
