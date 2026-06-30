"""Helpers for calling the OpenAI API to suggest a star schema.

This module is intentionally comment-heavy because the goal of the project is to
teach dimensional modeling concepts *inside* the code, not just in the README.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an authenticated OpenAI client.

    We load environment variables from `.env` so learners can keep secrets out of
    source control. In production, the same value would usually come from a
    secret manager or orchestrator environment variable instead.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to a .env file or run the pipeline with --skip-llm."
        )

    return OpenAI(api_key=api_key)


def suggest_star_schema(sample_data: str, column_info: str, client: OpenAI) -> dict[str, Any]:
    """Ask GPT-4o to propose a dimensional model from raw order data.

    Dimensional modeling is about organizing analytical data into facts and
    dimensions:
    - Dimensions describe business entities such as customers, products, and dates.
    - Facts capture measurable business events such as orders, clicks, or payments.
    - Grain describes the *exact* level of detail in the fact table. A grain of
      "one row per order line" is very different from "one row per customer per day".

    We request JSON because structured output is easier to validate, print,
    review, save for audit, and pass into downstream Python logic.
    """

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    system_prompt = """
You are a senior data warehouse architect and dimensional modeling expert.
Analyze the raw e-commerce order data and propose a clean star schema.

Return valid JSON with this exact top-level structure:
{
  "dimensions": [
    {
      "name": "dim_customer",
      "description": "...",
      "business_key": "customer_id",
      "columns": [
        {
          "target_column": "customer_name",
          "source_column": "customer_name",
          "role": "attribute"
        }
      ]
    }
  ],
  "fact_table": {
    "name": "fact_orders",
    "grain": "one row per ...",
    "columns": [
      {
        "target_column": "shipping_method",
        "source_column": "shipping_method",
        "role": "degenerate_dimension"
      }
    ],
    "measures": ["quantity", "unit_price", "discount_pct", "total_amount"],
    "foreign_keys": ["customer_key", "product_key", "date_key"]
  }
}

Be specific about source-to-target column mappings.
Prefer a practical star schema for analytics, not a normalized OLTP design.
Use dimensions that clearly separate descriptive attributes from measurable facts.
""".strip()

    user_prompt = f"""
Column profile:
{column_info}

Raw sample rows:
{sample_data}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The LLM returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse LLM JSON response: {exc}") from exc
