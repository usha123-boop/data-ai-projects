"""OpenAI client helpers for structured extraction.

This module is intentionally small because data engineers usually want one clear file where
all provider-specific logic lives. If you decide to move from OpenAI to Azure OpenAI, Bedrock,
or a local model later, this is the place to swap code while keeping the rest of the pipeline
unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Keeping the model name in a constant makes it obvious where to change it.
MODEL_NAME = "gpt-4o"

# This is a teaching estimate only. Real cost varies by model pricing and token usage.
ESTIMATED_COST_PER_1K_TOKENS_USD = 0.002


def get_client() -> OpenAI:
    """Create an OpenAI client using the API key stored in environment variables.

    We isolate client creation in a helper so notebooks, CLI runs, and future tests can all
    reuse the exact same setup logic.
    """
    # load_dotenv reads values from a local .env file into the current process.
    # That keeps API keys out of source control while still making local development easy.
    load_dotenv()

    # Reading configuration at runtime keeps this code portable across laptops, CI jobs,
    # containers, and orchestrators that inject environment variables dynamically.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    # The OpenAI client object is the main entry point for API calls.
    return OpenAI(api_key=api_key)


def extract_structured_with_client(
    client: OpenAI,
    text: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Call GPT-4o and return a parsed Python dictionary.

    Structured extraction means asking the model for named fields instead of a free-form answer.
    That matters for data engineering because columns like `sentiment` and `rating_prediction`
    are easy to filter, aggregate, test, and write to warehouses. Free-form prose is much harder
    to use downstream.
    """
    # We serialize the schema so the model sees the exact fields, allowed values, and data types
    # we expect. That makes outputs more consistent across many rows.
    schema_json = json.dumps(schema, indent=2)

    # The system prompt defines the high-level behavior of the model for this request.
    # We make the JSON-only requirement explicit because even one extra sentence can break a parser.
    system_prompt = f"""
    You are a precise data extraction engine.

    Return one valid JSON object and nothing else.
    Follow the requested schema exactly.
    If the review does not mention a value directly, infer the best answer from context.
    Keep `summary` to one sentence.
    Keep `key_issues` as a JSON array of short strings.
    Keep `rating_prediction` as an integer from 1 to 5.

    Target schema:
    {schema_json}
    """.strip()

    # response_format={"type": "json_object"} tells the API we want machine-friendly JSON.
    # This is safer than free-form text when the next step is `json.loads(...)`.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    # The SDK gives back the JSON payload as text, so we parse it immediately into a dict.
    message_content = response.choices[0].message.content or "{}"
    return json.loads(message_content)


def extract_structured(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper that creates the client for you.

    Why structured outputs instead of free-form text?
    - Analysts can group by exact labels like `positive` or `negative`.
    - Pipelines can validate field names and data types automatically.
    - Warehouses and BI tools work better with columns than paragraphs.

    How to swap providers later:
    - Azure OpenAI: replace `OpenAI(...)` with `AzureOpenAI(...)` and load endpoint/version vars.
    - Amazon Bedrock: replace the API call with a boto3 Bedrock invocation that returns the same dict.
    - Ollama or another local model: call the local endpoint but keep the return type unchanged.

    Cost note: for small extraction workloads, a rough planning estimate is about
    $0.002 per 1K tokens, though you should always verify current pricing.
    """
    client = get_client()
    return extract_structured_with_client(client=client, text=text, schema=schema)
