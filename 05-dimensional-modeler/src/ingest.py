"""Raw data ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_city",
    "customer_country",
    "product_id",
    "product_name",
    "product_category",
    "product_subcategory",
    "unit_price",
    "quantity",
    "discount_pct",
    "total_amount",
    "shipping_method",
    "order_status",
]


def load_orders(path: str | Path) -> pd.DataFrame:
    """Load the flat orders CSV into a DataFrame."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if df["order_date"].isna().any():
        raise ValueError("One or more order_date values could not be parsed.")

    numeric_columns = ["unit_price", "quantity", "discount_pct", "total_amount"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Column {column} contains non-numeric values.")

    return df


def get_data_profile(df: pd.DataFrame) -> str:
    """Create a compact profiling summary for the LLM.

    We deliberately send a profile plus a tiny sample instead of the full dataset.
    That is a common production pattern because:
    - it reduces token cost,
    - it limits exposure of raw data,
    - it speeds up the request,
    - and it still gives the model enough context to reason about likely facts,
      dimensions, keys, and measures.
    """

    lines: list[str] = []
    lines.append(f"Row count: {len(df):,}")
    lines.append(f"Column count: {len(df.columns)}")
    lines.append("")
    lines.append("Column-level profile:")

    for column in df.columns:
        series = df[column]
        non_null_values = series.dropna()
        unique_count = int(non_null_values.nunique())
        sample_values = [str(value) for value in non_null_values.astype(str).head(5).tolist()]
        lines.append(
            "- "
            f"{column}: dtype={series.dtype}, nulls={int(series.isna().sum())}, "
            f"unique={unique_count}, sample_values={sample_values}"
        )

    lines.append("")
    lines.append("Business interpretation hints:")
    lines.append("- Repeated customer_id values suggest a reusable customer dimension.")
    lines.append("- Repeated product_id values suggest a reusable product dimension.")
    lines.append("- order_date can drive a date dimension for time-based analysis.")
    lines.append("- quantity, unit_price, discount_pct, and total_amount behave like measures.")

    return "\n".join(lines)
