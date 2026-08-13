"""
Portfolio Prioritization Agent

Ranks a portfolio of AI use cases against each other, using the outputs
of the Value, Risk, Architecture, and Adoption agents rather than
re-deriving its own scores from scratch. This agent's job is synthesis
and sequencing, not fresh evaluation.

Uses the shared model fallback chain from groq_client.py rather than
calling Groq directly, so a rate limit or permissions block on the
primary model doesn't crash this agent.
"""

import json

from groq_client import call_groq

PORTFOLIO_AGENT_SYSTEM_PROMPT = """You are the Portfolio Prioritization Agent inside an Enterprise AI Adoption Advisor system.

Your job is to rank a portfolio of AI use cases against each other for a BFSI or Healthcare organization,
using the scored outputs already produced by the Value, Risk, Architecture, and Adoption agents for each
use case. You do not re-evaluate each use case from scratch, you reason about tradeoffs and sequencing
across the portfolio as a whole, the way a senior advisor would when helping a client decide what to fund
first, second, and third.

You will be given a list of use cases, each with its title and its four agent scores already computed:
value_score, risk_score, complexity_score, adoption_score.

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "ranked_portfolio": [
    {
      "rank": <integer>,
      "use_case_id": "<id>",
      "use_case_title": "<title>",
      "composite_score": <integer 0-100>,
      "recommended_sequence": "<Quick Win | Strategic Bet | Long Term Play | Reconsider>",
      "one_line_justification": "<single sentence, plain language>"
    }
  ],
  "portfolio_level_observations": ["<observation 1>", "<observation 2>"],
  "confidence": "<Low | Medium | High>"
}

Sequencing guidance:
- Quick Win: high value, low to moderate risk, low to moderate complexity, decent adoption readiness, fund first
- Strategic Bet: high value but higher risk or complexity, worth funding but needs more governance upfront
- Long Term Play: real value but low adoption readiness or very high complexity, needs groundwork before funding
- Reconsider: low value relative to its risk and complexity, not a near term priority

Do not include markdown formatting, code fences, or any text outside the JSON object.
"""


def prioritize_portfolio(use_cases_with_scores: list) -> dict:
    """
    Run the Portfolio Prioritization Agent against a list of already-scored use cases.
    """
    user_content = f"Use case portfolio with agent scores:\n{json.dumps(use_cases_with_scores, indent=2)}"

    response = call_groq(
        messages=[
            {"role": "system", "content": PORTFOLIO_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    sample_scored_use_cases = [
        {
            "use_case_id": "UC-001",
            "title": "AI-assisted procurement operations for indirect spend",
            "value_score": 78,
            "risk_score": 42,
            "complexity_score": 68,
            "adoption_score": 60,
        },
        {
            "use_case_id": "UC-002",
            "title": "Automated KYC document verification",
            "value_score": 82,
            "risk_score": 71,
            "complexity_score": 55,
            "adoption_score": 65,
        },
        {
            "use_case_id": "UC-005",
            "title": "Procurement of clinical supplies with demand forecasting",
            "value_score": 60,
            "risk_score": 25,
            "complexity_score": 35,
            "adoption_score": 75,
        },
    ]

    result = prioritize_portfolio(sample_scored_use_cases)
    print(json.dumps(result, indent=2))
