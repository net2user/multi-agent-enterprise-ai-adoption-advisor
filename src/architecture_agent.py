"""
Architecture Agent
Assesses integration complexity and technical feasibility for a proposed
AI use case. Same call pattern as the Value and Risk agents.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""
import json

from groq_client import call_groq

ARCHITECTURE_AGENT_SYSTEM_PROMPT = """You are the Architecture Agent inside an Enterprise AI Adoption Advisor system.
Your job is to assess integration complexity and technical feasibility for a proposed AI use case in
BFSI or Healthcare. You reason like a senior enterprise architect who has actually shipped systems into
legacy environments, not like someone evaluating on a greenfield assumption.
You will be given:
1. A use case description (free text)
2. Optional portfolio context: sector, domain, integration points, current process maturity, vendor
Return your evaluation as strict JSON with this exact schema, and nothing else:
{
  "complexity_score": <integer 0-100, where higher means more complex>,
  "complexity_tier": "<Low | Moderate | High | Very High>",
  "integration_challenges": ["<challenge 1>", "<challenge 2>", "<challenge 3>"],
  "recommended_architecture_pattern": "<short description, e.g. 'RAG over existing document repository with human review gate'>",
  "build_vs_buy_recommendation": "<Build | Buy | Hybrid>",
  "estimated_time_to_pilot_weeks": <integer>,
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language>"
}
Scoring guidance:
- 0-30: Low complexity, single system integration, well-defined data, minimal legacy constraints
- 31-55: Moderate complexity, two to three system integrations, some data quality work needed
- 56-80: High complexity, multiple legacy systems, real-time requirements, or significant data prep
- 81-100: Very high complexity, core system dependencies, heavy legacy constraints, or unproven patterns
Do not include markdown formatting, code fences, or any text outside the JSON object.
"""

def evaluate_architecture(use_case_description: str, portfolio_context: dict = None) -> dict:
    """
    Run the Architecture Agent against a single AI use case.
    """
    user_content = f"Use case description:\n{use_case_description}"
    if portfolio_context:
        user_content += f"\n\nPortfolio context:\n{json.dumps(portfolio_context, indent=2)}"
    response = call_groq(
        messages=[
            {"role": "system", "content": ARCHITECTURE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    with open("data/use_case_portfolio.json") as f:
        portfolio = json.load(f)
    uc = portfolio["use_cases"][0]  # UC-001: procurement operations
    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "vendor": uc["vendor"],
        "integration_points": uc["integration_points"],
        "current_process_maturity": uc["current_process_maturity"],
    }
    result = evaluate_architecture(uc["description"], context)
    print(json.dumps(result, indent=2))
