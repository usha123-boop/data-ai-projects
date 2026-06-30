"""Database setup helpers for the text-to-SQL project."""

from __future__ import annotations

from pathlib import Path

import duckdb


def setup_database(csv_path: str, db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Load the sales CSV into DuckDB and return a live connection.

    DuckDB is a great teaching tool for data engineers because it is:
    - embedded: no separate database server to install or manage
    - fast: it can scan local analytics data efficiently
    - SQL-friendly: its syntax feels familiar if you know warehouse SQL
    - close to real-world analytics engines: much of the query style mirrors Spark SQL,
      Databricks SQL, and other analytical databases

    For learning projects that need reproducibility, ``CREATE OR REPLACE TABLE`` is helpful
    because every run starts from the same CSV source of truth.
    """

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Load the CSV into a temporary in-memory connection first (needs filesystem access).
    # Then copy the data into the target connection which has external access disabled.
    # This two-step approach means user-supplied SQL queries can never read arbitrary files.
    loader = duckdb.connect(":memory:")
    loader.execute(
        """
        CREATE TABLE sales AS
        SELECT *
        FROM read_csv_auto(?, HEADER=TRUE)
        """,
        [str(csv_file)],
    )

    # Open the target connection with external filesystem/network access disabled.
    # Even if a malicious query slips through the guardrail checks, DuckDB will refuse
    # to open files or make network requests on this connection.
    conn = duckdb.connect(database=db_path, config={"enable_external_access": False})
    sales_df = loader.execute("SELECT * FROM sales").df()
    conn.register("sales_df", sales_df)
    conn.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM sales_df")
    conn.unregister("sales_df")
    loader.close()
    return conn


def get_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """Return a human-readable schema description for the sales table.

    LLMs do not have access to your database catalog unless you provide it. Converting the schema
    into a compact string gives the model the exact column spellings and types it must honor.
    That simple step dramatically reduces hallucinated columns and invalid SQL.
    """

    rows = conn.execute("DESCRIBE sales").fetchall()
    formatted_lines = ["sales table columns:"]
    for column_name, column_type, *_ in rows:
        formatted_lines.append(f"- {column_name}: {column_type}")
    return "\n".join(formatted_lines)
