"""Utilities for talking to OpenAI when generating SQL.

This module is intentionally small because we want the learning project to make the API
boundary obvious: prompt construction lives in ``src/generate_sql.py`` and the network call
lives here. In production you would often keep these concerns separate as well because it
makes prompt iteration safer and unit testing easier.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from src.generate_sql import build_system_prompt, extract_sql

DEFAULT_MODEL = "gpt-4o"


def get_client() -> OpenAI:
    """Create and return an authenticated OpenAI client.

    We load environment variables inside the helper so both the CLI and notebook can rely on
    the same setup step. Keeping this logic in one place also makes it easier to replace the
    provider later if your team moves from OpenAI to Azure OpenAI, Anthropic, or an internal
    gateway.
    """

    # Loading .env inside the helper keeps the notebook and CLI setup identical.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key first."
        )

    return OpenAI(api_key=api_key)


def generate_sql(question: str, schema: str, client: OpenAI) -> str:
    """Ask the LLM to convert a natural-language analytics question into SQL.

    The model performs much better when we include two things in the prompt:
    1. The *current schema* so it knows the exact column names and data types.
    2. A few examples of good input/output pairs so it can imitate the pattern.

    This is the core few-shot prompting idea: instead of fine-tuning a new model, we show the
    model what a correct answer looks like. As your SQL tasks grow more complex, adding more
    domain-specific examples (dates, joins, filters, ranking, window functions) usually gives a
    bigger quality improvement than vague prompt wording alone.
    """

    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    system_prompt = build_system_prompt(schema)

    try:
        response = client.chat.completions.create(
            model=model_name,
            # A low temperature makes SQL generation more deterministic and reviewable.
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Convert the following analytics question into SQL for the sales table. "
                        "Return SQL only.\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    message_content: Any = response.choices[0].message.content
    if not message_content:
        raise RuntimeError("The model returned an empty response.")

    return extract_sql(str(message_content))
