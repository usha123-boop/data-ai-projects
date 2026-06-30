"""Command-line entry point for the LLM enrichment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .enrich import EMPTY_EXTRACTION, enrich_batch
from .ingest import load_reviews
from .llm import get_client


def run(input_path: str, output_path: str, sample: int | None = None) -> pd.DataFrame:
    """Run the full pipeline from raw CSV to enriched Parquet.

    This function intentionally mirrors a classic ETL flow: ingest, transform, and write.
    The only difference is that the transform step includes an LLM call.
    """
    # Loading .env here makes the CLI friendlier because users can run the module directly.
    load_dotenv()

    print("🚀 Starting LLM enrichment pipeline")

    # Step 1: load and validate the input before any paid model requests happen.
    df = load_reviews(input_path)

    # Step 2: create one reusable client for the whole batch.
    client = get_client()

    # Step 3: enrich the data with new structured columns.
    enriched_df = enrich_batch(df=df, client=client, sample=sample)

    # Step 4: create the output directory if it does not exist yet.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Step 5: save Parquet because it preserves types and works well with analytics engines.
    enriched_df.to_parquet(output_file, index=False)

    added_columns = list(EMPTY_EXTRACTION.keys())
    print("✅ Pipeline finished successfully")
    print(f"📦 Rows processed: {len(enriched_df)}")
    print(f"🧠 Columns added: {len(added_columns)} -> {', '.join(added_columns)}")
    print(f"📝 Output path: {output_file}")
    return enriched_df


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for `python -m src.pipeline`."""
    parser = argparse.ArgumentParser(description="Run the LLM enrichment pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to the output Parquet file")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows to process from the top of the file",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(input_path=args.input, output_path=args.output, sample=args.sample)
