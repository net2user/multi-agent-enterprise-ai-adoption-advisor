"""
Enterprise AI Adoption Advisor - Agent Definitions
Business Value Agent

Grounds its output in retrieved RBI regulatory text via the
ai-adoption-rag-core API, falling back to ungrounded output if that
service isn't reachable.

Now uses the shared model fallback chain from groq_client.py, same as
every other agent, instead of its own separate inline copy.
"""

import json
import os
import requests

from groq_client import call_groq

RAG_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")

VALUE_AGENT_SYSTEM_PROMPT = """You are the Business Value Agent inside an Enterprise AI Adoption Advisor system.

Your job is to evaluate the expected business impact of a proposed AI use case for BFSI or Healthcare
organizations. You reason like a senior digital transformation advisor with deep operational experience,
not like a generic assistant. You are specific, you quantify where you can, and you do not inflate value
estimates to sound impressive.

You will be given:
1. A use case description (free text)
2. Optional portfolio context: cost, sector, domain, current process maturity
3. Optional retrieved regulatory context: excerpts from actual RBI source documents relevant to this use case

When retrieved regulatory context is provided, let it inform value drivers tied to compliance efficiency,
reduced audit burden, or avoided regulatory penalty, referencing what the source document actually says
rather than general knowledge. If a retrieved excerpt is marked as draft status, phrase anything drawn from
it as "the draft guidance proposes" or similar, never as "RBI requires," since draft guidance is not yet binding.

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "value_score": <integer 0-100>,
  "value_tier": "<Low | Moderate | High | Transformational>",
  "estimated_annual_value_range_usd": "<string range, e.g. '1.2M - 1.8M'>",
  "value_drivers": ["<driver 1>", "<driver 2>", "<driver 3>"],
  "key_assumptions": ["<assumption 1>", "<assumption 2>"],
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language>"
}

Each value driver must include a short justification, an estimated magnitude, percentage, or concrete mechanism,
not just a category name. Write "Reduced pricing anomalies through AI-driven contract review, an estimated 10-15
percent reduction in erroneous payments" rather than just "Reduced pricing anomalies through AI-driven review."
If a precise figure cannot be reasonably inferred, name the specific mechanism driving the value instead of a
bare category label.

Scoring guidance:
- 0-30: Low value, likely a point solution or process convenience only
- 31-55: Moderate value, meaningful efficiency gain in one function
- 56-80: High value, measurable impact on cost, revenue, or risk exposure at a business unit level
- 81-100: Transformational, impact spans multiple functions or changes a core operating model

Do not include markdown formatting, code fences, or any text outside the JSON object.
"""


def retrieve_context(query: str, n_results: int = 3) -> list:
    try:
        response = requests.post(
            f"{RAG_API_URL}/retrieve",
            json={"query": query, "n_results": n_results},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Warning: RAG retrieval unavailable ({e}). Proceeding without grounding.")
        return []


def format_retrieved_context(chunks: list) -> str:
    if not chunks:
        return ""
    parts = ["Retrieved regulatory context:"]
    for c in chunks:
        parts.append(
            f"\n[Source: {c['title']} | File: {c['file']} | Status: {c['status']}]\n{c['text']}"
        )
    return "\n".join(parts)


def evaluate_business_value(use_case_description: str, portfolio_context: dict = None) -> dict:
    user_content = f"Use case description:\n{use_case_description}"
    if portfolio_context:
        user_content += f"\n\nPortfolio context:\n{json.dumps(portfolio_context, indent=2)}"

    retrieved = retrieve_context(use_case_description)
    context_block = format_retrieved_context(retrieved)
    if context_block:
        user_content += f"\n\n{context_block}"

    response = call_groq(
        messages=[
            {"role": "system", "content": VALUE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


if __name__ == "__main__":
    with open("data/use_case_portfolio.json") as f:
        portfolio = json.load(f)

    uc = portfolio["use_cases"][0]
    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "estimated_annual_cost_usd": uc["estimated_annual_cost_usd"],
        "current_process_maturity": uc["current_process_maturity"],
    }

    result = evaluate_business_value(uc["description"], context)
    print(json.dumps(result, indent=2))
