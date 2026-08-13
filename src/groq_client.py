"""
Shared Groq client and model fallback chain.

Every agent that calls Groq should import client and call_groq from
here instead of building its own OpenAI client and model list. This
is the one place the fallback chain lives, so a rate limit or
permissions fix only needs to happen once, not per agent file.

Tries each model in MODEL_CANDIDATES in order, falling back to the
next on either a rate limit (429) or a permissions block (403), since
both mean "this model isn't usable right now, try the next one."
"""

import os
from openai import OpenAI, RateLimitError, PermissionDeniedError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL_CANDIDATES = os.environ.get(
    "GROQ_MODEL_CANDIDATES",
    "openai/gpt-oss-120b,groq/compound-mini,groq/compound,qwen/qwen3.6-27b,openai/gpt-oss-20b,llama-3.3-70b-versatile,llama-3.1-8b-instant"
).split(",")


def call_groq(messages, temperature):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.chat.completions.create(
                model=model_name.strip(),
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            if model_name.strip() != MODEL_CANDIDATES[0].strip():
                print(f"Note: succeeded using fallback model {model_name.strip()}")
            return response
        except (RateLimitError, PermissionDeniedError) as e:
            print(f"Warning: {model_name.strip()} unavailable ({type(e).__name__}), trying next candidate...")
            last_error = e
            continue
    raise last_error
