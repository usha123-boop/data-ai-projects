"""LLM-powered enrichment functions for review text."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from . import llm

# This schema is intentionally business-friendly.
# Every field maps naturally to a column that analysts can group, filter, or chart.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "sentiment": {
        "type": "string",
        "allowed_values": ["positive", "negative", "neutral"],
    },
    "category": {
        "type": "string",
        "allowed_values": ["quality", "shipping", "price", "support", "other"],
    },
    "key_issues": {
        "type": "array",
        "items": "string",
        "description": "Important issues, complaints, or notable themes from the review.",
    },
    "summary": {
        "type": "string",
        "description": "One-sentence business summary of the review.",
    },
    "rating_prediction": {
        "type": "integer",
        "allowed_values": [1, 2, 3, 4, 5],
    },
}

# When an API call fails, we still want a row-shaped record so concatenation keeps working.
EMPTY_EXTRACTION: dict[str, Any] = {
    "sentiment": None,
    "category": None,
    "key_issues": None,
    "summary": None,
    "rating_prediction": None,
}

# This rough estimate helps learners think about cost before running large jobs.
ESTIMATED_TOKENS_PER_ROW = 500


def build_extraction_prompt(review_text: str) -> str:
    """Create the request text that guides the model toward a tabular answer.

    For data engineers, the important idea is turning a fuzzy business question into explicit
    fields. Once the request names the output columns clearly, the LLM becomes an enrichment
    step rather than a generic chatbot.
    """
    return f"""
    Read the e-commerce review below and extract structured fields.

    Return values for:
    - sentiment: one of positive, negative, neutral
    - category: one of quality, shipping, price, support, other
    - key_issues: list of short strings
    - summary: one sentence summarizing the review
    - rating_prediction: integer from 1 to 5

    Review text:
    {review_text}
    """.strip()


def _normalize_extraction(result: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw model output into a stable shape for DataFrame columns.

    Models are probabilistic, so even when we ask for JSON they may vary in casing or types.
    This normalization layer makes downstream analytics much more predictable.
    """
    normalized = EMPTY_EXTRACTION.copy()

    sentiment = result.get("sentiment")
    if isinstance(sentiment, str) and sentiment.lower() in {"positive", "negative", "neutral"}:
        normalized["sentiment"] = sentiment.lower()

    category = result.get("category")
    if isinstance(category, str) and category.lower() in {"quality", "shipping", "price", "support", "other"}:
        normalized["category"] = category.lower()

    key_issues = result.get("key_issues")
    if isinstance(key_issues, list):
        normalized["key_issues"] = [str(item).strip() for item in key_issues if str(item).strip()]
    elif isinstance(key_issues, str) and key_issues.strip():
        normalized["key_issues"] = [key_issues.strip()]

    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        normalized["summary"] = summary.strip()

    rating_prediction = result.get("rating_prediction")
    if isinstance(rating_prediction, (int, float)) and not math.isnan(float(rating_prediction)):
        rating_as_int = int(rating_prediction)
        if rating_as_int in {1, 2, 3, 4, 5}:
            normalized["rating_prediction"] = rating_as_int

    return normalized


def enrich_single(review_text: str, client: OpenAI) -> dict[str, Any]:
    """Enrich one review with structured fields from the LLM."""
    # We build the prompt separately so it is easy to inspect and tweak during experimentation.
    prompt = build_extraction_prompt(review_text)

    # The LLM wrapper handles provider-specific details and returns a Python dict.
    raw_result = llm.extract_structured_with_client(
        client=client,
        text=prompt,
        schema=EXTRACTION_SCHEMA,
    )

    # Normalization gives the rest of the pipeline a stable shape even if the model is imperfect.
    return _normalize_extraction(raw_result)


def enrich_batch(df: pd.DataFrame, client: OpenAI, sample: int | None = None) -> pd.DataFrame:
    """Loop through reviews, enrich each row, and append new columns.

    This is the core data engineering pattern in the project: iterate through records, enrich
    them one at a time, capture failures per row, and produce a clean table at the end.
    """
    # We copy the input DataFrame so the function is easier to reason about and does not mutate
    # the caller's object unexpectedly.
    working_df = df.copy()

    # Sampling is a practical cost-control feature when you are testing prompts.
    if sample is not None:
        if sample <= 0:
            raise ValueError("sample must be a positive integer when provided")
        working_df = working_df.head(sample).copy()
        print(f"🔎 Sampling enabled: enriching first {len(working_df)} rows")

    # A rough estimate helps users decide whether to run a full batch or a sample first.
    estimated_cost = (
        (len(working_df) * ESTIMATED_TOKENS_PER_ROW / 1000)
        * llm.ESTIMATED_COST_PER_1K_TOKENS_USD
    )
    print(f"💸 Rough estimated LLM cost for this run: ${estimated_cost:.4f}")

    enrichment_records: list[dict[str, Any]] = []

    # tqdm adds a progress bar, which is helpful because network-backed row processing can take time.
    for row in tqdm(
        working_df.itertuples(index=False),
        total=len(working_df),
        desc="Enriching reviews",
    ):
        try:
            enrichment_records.append(enrich_single(review_text=row.review_text, client=client))
        except Exception as exc:
            # A resilient pipeline records the failure, logs context, and moves on to the next row.
            print(f"⚠️ Failed to enrich review {row.review_id}: {exc}")
            enrichment_records.append(EMPTY_EXTRACTION.copy())

    # Turning the list of dicts into a DataFrame gives us one column per extracted field.
    enriched_columns = pd.DataFrame(enrichment_records)

    # reset_index keeps row alignment correct before concatenating the new columns.
    enriched_df = pd.concat([working_df.reset_index(drop=True), enriched_columns], axis=1)
    print(f"✨ Added {len(enriched_columns.columns)} enrichment columns")
    return enriched_df
