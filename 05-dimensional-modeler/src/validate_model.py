"""Validation helpers for the dimensional model."""

from __future__ import annotations

from typing import Any

import pandas as pd


# Referential integrity matters because analytical joins assume every foreign key
# in the fact table has a matching row in the related dimension. If that link is
# broken, dashboards silently undercount, show NULL buckets, or attribute revenue
# to the wrong entities. Warehouses should validate these assumptions just like
# application teams validate API contracts.


def _check_fk(fact_df: pd.DataFrame, fact_key: str, dim_df: pd.DataFrame, dim_key: str) -> dict[str, Any]:
    missing_mask = ~fact_df[fact_key].isin(dim_df[dim_key])
    missing_count = int(missing_mask.sum())
    return {
        "fact_key": fact_key,
        "dimension_key": dim_key,
        "checked_rows": int(len(fact_df)),
        "missing_rows": missing_count,
        "passed": missing_count == 0,
    }


def validate_referential_integrity(
    fact_df: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> dict[str, Any]:
    """Validate that all fact foreign keys exist in their dimensions."""

    results = {
        "customer_fk": _check_fk(fact_df, "customer_key", dim_customer, "customer_key"),
        "product_fk": _check_fk(fact_df, "product_key", dim_product, "product_key"),
        "date_fk": _check_fk(fact_df, "date_key", dim_date, "date_key"),
    }
    results["all_passed"] = all(section["passed"] for section in results.values())
    return results


def print_validation_report(results: dict[str, Any]) -> None:
    """Display validation results in a human-friendly format."""

    print("\n✅ Referential Integrity Report")
    print("=" * 80)
    for section_name, section in results.items():
        if section_name == "all_passed":
            continue
        emoji = "✅" if section["passed"] else "❌"
        print(
            f"{emoji} {section_name}: checked={section['checked_rows']}, "
            f"missing={section['missing_rows']}, passed={section['passed']}"
        )

    overall_emoji = "🎉" if results["all_passed"] else "⚠️"
    print(f"{overall_emoji} Overall result: {results['all_passed']}")
