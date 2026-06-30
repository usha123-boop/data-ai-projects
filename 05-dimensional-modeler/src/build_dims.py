"""Dimension table builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Surrogate keys are warehouse-generated integer keys such as customer_key.
# We use them instead of business/natural keys for a few reasons:
# 1. they are compact and join efficiently,
# 2. they isolate the warehouse from source-system key changes,
# 3. they make slowly changing dimension (SCD) handling easier when attributes
#    such as customer city or product category change over time.
# In this tutorial we build simple type-1 style dimensions, but the same idea
# extends to more advanced SCD patterns used in production warehouses.


def build_dim_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Build a deduplicated customer dimension."""

    customer_columns = [
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_city",
        "customer_country",
    ]
    dim_customer = (
        df[customer_columns]
        .drop_duplicates(subset=["customer_id"])
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    dim_customer.insert(0, "customer_key", range(1, len(dim_customer) + 1))
    return dim_customer


def build_dim_product(df: pd.DataFrame) -> pd.DataFrame:
    """Build a deduplicated product dimension.

    Note: unit_price is intentionally excluded from this dimension.
    Prices can change over time (promotions, repricing), so storing any single
    price on the dimension would be non-deterministic (first row wins) and
    misleading for analysts. The per-order price is already captured as a
    measure in fact_orders — that is the correct and authoritative place for it.
    """

    product_columns = [
        "product_id",
        "product_name",
        "product_category",
        "product_subcategory",
    ]
    dim_product = (
        df[product_columns]
        .drop_duplicates(subset=["product_id"])
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    dim_product.insert(0, "product_key", range(1, len(dim_product) + 1))
    return dim_product


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical date dimension.

    Every warehouse benefits from a date dimension because analysts constantly
    ask questions by year, quarter, month, week, weekday, and weekend/weekday.
    Precomputing those attributes avoids repeating date logic across hundreds of
    dashboards and SQL queries.
    """

    unique_dates = (
        pd.DataFrame({"date": pd.to_datetime(df["order_date"]).dt.normalize().drop_duplicates()})
        .sort_values("date")
        .reset_index(drop=True)
    )

    iso_calendar = unique_dates["date"].dt.isocalendar()
    unique_dates["date_key"] = unique_dates["date"].dt.strftime("%Y%m%d").astype(int)
    unique_dates["year"] = unique_dates["date"].dt.year
    unique_dates["quarter"] = unique_dates["date"].dt.quarter
    unique_dates["month"] = unique_dates["date"].dt.month
    unique_dates["month_name"] = unique_dates["date"].dt.strftime("%B")
    unique_dates["week"] = iso_calendar.week.astype(int)
    unique_dates["day_of_week"] = unique_dates["date"].dt.dayofweek
    unique_dates["day_name"] = unique_dates["date"].dt.strftime("%A")
    unique_dates["is_weekend"] = unique_dates["day_of_week"] >= 5

    ordered_columns = [
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "week",
        "day_of_week",
        "day_name",
        "is_weekend",
    ]
    return unique_dates[ordered_columns]


def save_dimensions(dims: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Save dimension tables as Parquet files."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_df in dims.items():
        output_path = target_dir / f"{table_name}.parquet"
        table_df.to_parquet(output_path, index=False)
        print(f"💾 Saved {table_name} ({len(table_df):,} rows) to {output_path}")
