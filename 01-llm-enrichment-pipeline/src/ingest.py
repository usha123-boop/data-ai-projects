"""CSV loading and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# These are the columns the rest of the pipeline depends on.
REQUIRED_COLUMNS = {"review_id", "product_name", "review_text", "date"}


def load_reviews(path: str) -> pd.DataFrame:
    """Load the raw review CSV and validate the minimum data contract.

    Validation matters because LLM calls are the expensive part of the pipeline. It is better to
    catch a broken file immediately than to discover a missing column after paying for API usage.
    """
    # Path gives us a reliable cross-platform way to work with file locations.
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")

    # pandas is a standard tool for tabular ingestion and keeps the example approachable.
    df = pd.read_csv(csv_path)

    # Informative print statements help learners confirm the pipeline is reading the file they expect.
    print(f"✅ Loaded {len(df)} reviews from {csv_path}")
    print(f"📐 Raw shape: {df.shape}")

    # Missing columns usually mean the upstream export changed. We fail fast with a clear message.
    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Expected columns: {sorted(REQUIRED_COLUMNS)}"
        )

    # A syntactically valid CSV can still be useless if it has no rows.
    if df.empty:
        raise ValueError("The input CSV is empty. Add at least one review row before running.")

    # review_id behaves like a primary key for this small dataset.
    # Duplicate IDs are usually a sign of ingestion mistakes or accidental double loads.
    if df["review_id"].duplicated().any():
        duplicate_ids = df.loc[df["review_id"].duplicated(), "review_id"].tolist()
        raise ValueError(f"Duplicate review_id values found: {duplicate_ids}")

    # We convert dates once here so later steps do not need to guess whether this column is text.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df.loc[df["date"].isna(), "review_id"].tolist()
        raise ValueError(f"Invalid dates found for review_id values: {bad_rows}")

    # Casting text columns to string avoids edge cases from nulls or mixed CSV types.
    for column_name in ["review_id", "product_name", "review_text"]:
        df[column_name] = df[column_name].astype(str).str.strip()

    print(f"🧾 Validated columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
    return df
