"""End-to-end pipeline for LLM-assisted dimensional modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .build_dims import build_dim_customer, build_dim_date, build_dim_product, save_dimensions
from .build_facts import build_fact_orders, save_fact_table
from .ingest import get_data_profile, load_orders
from .llm import get_client
from .suggest_schema import (
    analyze_and_suggest,
    get_default_schema,
    print_schema_proposal,
    save_schema_proposal,
)
from .validate_model import print_validation_report, validate_referential_integrity


def run_pipeline(input_path: str, output_dir: str, skip_llm: bool = False) -> dict[str, Any]:
    """Run the full workflow from raw CSV to validated star schema outputs."""

    print("🚀 Starting the LLM-assisted dimensional modeling pipeline")
    print(f"📥 Input file : {input_path}")
    print(f"📤 Output dir : {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n1️⃣ Loading raw orders...")
    df = load_orders(input_path)
    print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")

    print("\n2️⃣ Profiling data for schema design...")
    profile = get_data_profile(df)
    print(profile)

    print("\n3️⃣ Proposing a star schema...")
    if skip_llm:
        print("🛟 --skip-llm enabled. Using the hardcoded fallback schema.")
        schema = get_default_schema()
    else:
        try:
            client = get_client()
            schema = analyze_and_suggest(df, client)
        except Exception as exc:
            raise RuntimeError(
                "LLM schema suggestion failed. Check your API key or rerun with --skip-llm."
            ) from exc

    print_schema_proposal(schema)
    save_schema_proposal(schema, output_path / "schema_proposal.json")

    print("\n4️⃣ Building dimensions...")
    dim_customer = build_dim_customer(df)
    dim_product = build_dim_product(df)
    dim_date = build_dim_date(df)
    dims = {
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_date": dim_date,
    }
    save_dimensions(dims, output_path)

    print("\n5️⃣ Building fact table...")
    fact_df = build_fact_orders(df, dim_customer, dim_product, dim_date)
    save_fact_table(fact_df, output_path)

    print("\n6️⃣ Validating model integrity...")
    validation_results = validate_referential_integrity(
        fact_df=fact_df,
        dim_customer=dim_customer,
        dim_product=dim_product,
        dim_date=dim_date,
    )
    print_validation_report(validation_results)

    print("\n7️⃣ Pipeline summary")
    print("=" * 80)
    print(f"👥 dim_customer rows: {len(dim_customer):,}")
    print(f"📦 dim_product rows : {len(dim_product):,}")
    print(f"🗓️  dim_date rows    : {len(dim_date):,}")
    print(f"🧾 fact_orders rows : {len(fact_df):,}")
    print(f"📁 Output written to: {output_path.resolve()}")

    return {
        "schema": schema,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_date": dim_date,
        "fact_orders": fact_df,
        "validation": validation_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-assisted dimensional modeling pipeline")
    parser.add_argument("--input", required=True, help="Path to the raw orders CSV")
    parser.add_argument("--output", required=True, help="Directory for parquet outputs")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Use a hardcoded schema instead of calling the OpenAI API",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(input_path=args.input, output_dir=args.output, skip_llm=args.skip_llm)
