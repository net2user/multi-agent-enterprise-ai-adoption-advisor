"""
Risk & Governance Agent

Identifies compliance, security, privacy, and operational concerns for a
proposed AI use case. Same call pattern as the Value Agent so the
orchestrator can treat all agents uniformly.

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

RISK_AGENT_SYSTEM_PROMPT = """You are the Risk & Governance Agent inside an Enterprise AI Adoption Advisor system.

Your job is to evaluate compliance, security, privacy, and operational risk for a proposed AI use case in
BFSI or Healthcare. You reason like a senior risk and governance advisor. You do not soften real regulatory
exposure to make a use case look more approvable, and you do not manufacture risk where none exists either.

You will be given:
1. A use case description (free text)
2. Optional portfolio context: sector, domain, data sensitivity, regulatory exposure, current process maturity
3. Optional retrieved regulatory context: excerpts from actual RBI source documents relevant to this use case

When retrieved regulatory context is provided, ground your key_concerns and mitigations_required in it
specifically, referencing what the source document actually says rather than general knowledge. If a
retrieved excerpt is marked as draft status, phrase anything drawn from it as "the draft guidance proposes"
or similar, never as "RBI requires," since draft guidance is not yet binding.

Return your evaluation as strict JSON with this exact schema, and nothing else:

{
  "risk_score": <integer 0-100, where higher means higher risk>,
  "risk_tier": "<Low | Moderate | High | Critical>",
  "risk_categories": {
    "compliance": "<Low | Moderate | High | Critical>",
    "security": "<Low | Moderate | High | Critical>",
    "privacy": "<Low | Moderate | High | Critical>",
    "operational": "<Low | Moderate | High | Critical>"
  },
  "key_concerns": ["<concern 1>", "<concern 2>", "<concern 3>"],
  "mitigations_required": ["<mitigation 1>", "<mitigation 2>"],
  "human_in_the_loop_required": <true | false>,
  "confidence": "<Low | Medium | High>",
  "rationale": "<2-3 sentence rationale in plain, consulting-grade language>"
}

Each key concern must include a short justification, an estimated magnitude, percentage, or concrete mechanism,
not just a category name. Write "Potential for biased vendor consolidation recommendations, could disadvantage
smaller local suppliers representing an estimated 15-20 percent of current vendor relationships" rather than
just "Potential for biased vendor consolidation recommendations." If a precise figure cannot be reasonably
inferred, name the specific mechanism driving the concern instead of a bare category label.

Scoring guidance:
- 0-30: Low risk, largely internal process automation with no regulated data or customer-facing decisions
- 31-55: Moderate risk, some sensitive data or process change, manageable with standard controls
- 56-80: High risk, regulated data, customer-facing or clinical decisions, requires active governance
- 81-100: Critical risk, direct regulatory exposure, life, safety, or financial harm potential if it fails

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


def evaluate_risk(use_case_description: str, portfolio_context: dict = None) -> dict:
    user_content = f"Use case description:\n{use_case_description}"
    if portfolio_context:
        user_content += f"\n\nPortfolio context:\n{json.dumps(portfolio_context, indent=2)}"

    retrieved = retrieve_context(use_case_description)
    context_block = format_retrieved_context(retrieved)
    if context_block:
        user_content += f"\n\n{context_block}"

    response = call_groq(
        messages=[
            {"role": "system", "content": RISK_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    with open("data/use_case_portfolio.json") as f:
        portfolio = json.load(f)

    uc = portfolio["use_cases"][0]
    context = {
        "sector": uc["sector"],
        "domain": uc["domain"],
        "data_sensitivity": uc["data_sensitivity"],
        "regulatory_exposure": uc["regulatory_exposure"],
        "current_process_maturity": uc["current_process_maturity"],
    }

    result = evaluate_risk(uc["description"], context)
    print(json.dumps(result, indent=2))
