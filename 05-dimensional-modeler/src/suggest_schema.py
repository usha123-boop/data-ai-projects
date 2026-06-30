"""Analyze raw data and propose a dimensional model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import get_data_profile
from .llm import suggest_star_schema


# This fallback schema is intentionally explicit so the project still runs when a
# learner does not yet have an API key. In real projects, teams often combine an
# LLM draft with a deterministic guardrail like this before any production use.
DEFAULT_SCHEMA: dict[str, Any] = {
    "dimensions": [
        {
            "name": "dim_customer",
            "description": "Customer dimension containing descriptive customer attributes.",
            "business_key": "customer_id",
            "columns": [
                {"target_column": "customer_id", "source_column": "customer_id", "role": "business_key"},
                {"target_column": "customer_name", "source_column": "customer_name", "role": "attribute"},
                {"target_column": "customer_email", "source_column": "customer_email", "role": "attribute"},
                {"target_column": "customer_city", "source_column": "customer_city", "role": "attribute"},
                {"target_column": "customer_country", "source_column": "customer_country", "role": "attribute"},
            ],
        },
        {
            "name": "dim_product",
            "description": "Product dimension for slicing order metrics by product hierarchy.",
            "business_key": "product_id",
            "columns": [
                {"target_column": "product_id", "source_column": "product_id", "role": "business_key"},
                {"target_column": "product_name", "source_column": "product_name", "role": "attribute"},
                {"target_column": "product_category", "source_column": "product_category", "role": "attribute"},
                {"target_column": "product_subcategory", "source_column": "product_subcategory", "role": "attribute"},
                {"target_column": "unit_price", "source_column": "unit_price", "role": "attribute"},
            ],
        },
        {
            "name": "dim_date",
            "description": "Calendar dimension used for consistent time-based reporting.",
            "business_key": "order_date",
            "columns": [
                {"target_column": "date", "source_column": "order_date", "role": "business_key"},
                {"target_column": "year", "source_column": "derived", "role": "attribute"},
                {"target_column": "quarter", "source_column": "derived", "role": "attribute"},
                {"target_column": "month", "source_column": "derived", "role": "attribute"},
                {"target_column": "month_name", "source_column": "derived", "role": "attribute"},
                {"target_column": "week", "source_column": "derived", "role": "attribute"},
                {"target_column": "day_of_week", "source_column": "derived", "role": "attribute"},
                {"target_column": "day_name", "source_column": "derived", "role": "attribute"},
                {"target_column": "is_weekend", "source_column": "derived", "role": "attribute"},
            ],
        },
    ],
    "fact_table": {
        "name": "fact_orders",
        "grain": "one row per order record in the raw input",
        "columns": [
            {"target_column": "shipping_method", "source_column": "shipping_method", "role": "degenerate_dimension"},
            {"target_column": "order_status", "source_column": "order_status", "role": "degenerate_dimension"},
        ],
        "measures": ["quantity", "unit_price", "discount_pct", "total_amount"],
        "foreign_keys": ["customer_key", "product_key", "date_key"],
    },
}


def analyze_and_suggest(df: pd.DataFrame, client: Any) -> dict[str, Any]:
    """Profile the raw data and ask the LLM to draft a star schema.

    A star schema keeps business events in the center (facts) and descriptive
    context around the outside (dimensions). This pattern is popular because it
    makes BI tools, SQL joins, and aggregation logic simpler for analysts.
    """

    profile = get_data_profile(df)
    sample_data = df.head(10).to_csv(index=False)
    return suggest_star_schema(sample_data=sample_data, column_info=profile, client=client)


def get_default_schema() -> dict[str, Any]:
    """Return a deterministic schema for offline runs and tests."""

    return DEFAULT_SCHEMA


def print_schema_proposal(schema: dict[str, Any]) -> None:
    """Pretty-print the proposed star schema."""

    print("\n🧠 Proposed Star Schema")
    print("=" * 80)

    for dimension in schema.get("dimensions", []):
        print(f"\n📦 Dimension: {dimension['name']}")
        print(f"   Description : {dimension.get('description', 'n/a')}")
        print(f"   Business key: {dimension.get('business_key', 'n/a')}")
        for column in dimension.get("columns", []):
            print(
                "   - "
                f"{column.get('target_column')} <= {column.get('source_column')} "
                f"({column.get('role')})"
            )

    fact_table = schema.get("fact_table", {})
    print(f"\n📊 Fact Table: {fact_table.get('name', 'unknown')}")
    print(f"   Grain       : {fact_table.get('grain', 'n/a')}")
    print(f"   Measures    : {', '.join(fact_table.get('measures', []))}")
    print(f"   Foreign keys: {', '.join(fact_table.get('foreign_keys', []))}")
    for column in fact_table.get("columns", []):
        print(
            "   - "
            f"{column.get('target_column')} <= {column.get('source_column')} "
            f"({column.get('role')})"
        )


def save_schema_proposal(schema: dict[str, Any], path: str | Path) -> None:
    """Persist the schema proposal for audit, review, and reproducibility."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(schema, file_handle, indent=2)
    print(f"\n💾 Saved schema proposal to {output_path}")
