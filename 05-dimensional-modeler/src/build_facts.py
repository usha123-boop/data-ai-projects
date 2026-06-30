"""Fact table builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# The grain of this fact table is one row per raw order record in the source
# file. Grain is the single most important design choice in a fact table because
# it determines which aggregations are valid.
#
# Measures are numeric values we want to aggregate, such as quantity and revenue.
# Foreign keys point to surrounding dimensions so analysts can slice those
# measures by customer, product, and time.
#
# We keep surrogate keys in the fact table because they are warehouse-native and
# stable. Natural keys like customer_id are still useful for lineage, but they
# belong in dimensions rather than in every fact row.


def build_fact_orders(
    df: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> pd.DataFrame:
    """Build the central fact table from raw orders plus dimensions."""

    fact_df = df.copy()
    fact_df["order_date"] = pd.to_datetime(fact_df["order_date"]).dt.normalize()

    fact_df = fact_df.merge(
        dim_customer[["customer_id", "customer_key"]],
        on="customer_id",
        how="left",
    )
    fact_df = fact_df.merge(
        dim_product[["product_id", "product_key"]],
        on="product_id",
        how="left",
    )
    fact_df = fact_df.merge(
        dim_date[["date", "date_key"]],
        left_on="order_date",
        right_on="date",
        how="left",
    )

    missing_keys = {
        "customer_key": int(fact_df["customer_key"].isna().sum()),
        "product_key": int(fact_df["product_key"].isna().sum()),
        "date_key": int(fact_df["date_key"].isna().sum()),
    }
    if any(count > 0 for count in missing_keys.values()):
        raise ValueError(f"Failed to resolve surrogate keys for fact table: {missing_keys}")

    fact_df = fact_df.sort_values("order_id").reset_index(drop=True)
    fact_df.insert(0, "order_key", range(1, len(fact_df) + 1))

    final_columns = [
        "order_key",
        "customer_key",
        "product_key",
        "date_key",
        "quantity",
        "unit_price",
        "discount_pct",
        "total_amount",
        "shipping_method",
        "order_status",
    ]
    fact_df = fact_df[final_columns]

    integer_columns = ["order_key", "customer_key", "product_key", "date_key", "quantity"]
    for column in integer_columns:
        fact_df[column] = fact_df[column].astype(int)

    return fact_df


def save_fact_table(fact_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save the fact table as Parquet."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "fact_orders.parquet"
    fact_df.to_parquet(output_path, index=False)
    print(f"💾 Saved fact_orders ({len(fact_df):,} rows) to {output_path}")
