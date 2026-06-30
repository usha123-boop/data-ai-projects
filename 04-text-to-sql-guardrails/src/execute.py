"""Execution helpers for validated SQL queries."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from src.validate import run_guardrails


def execute_query(sql: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Execute a validated query and return the results as a pandas DataFrame."""

    return conn.execute(sql).fetch_df()


def format_results(df: pd.DataFrame) -> str:
    """Format query results for clean terminal output.

    Terminal demos are much easier to follow when the output is predictable and compact. Returning
    a string also keeps the CLI logic simple because notebooks can use the DataFrame directly while
    the command-line pipeline can print the formatted representation.
    """

    if df.empty:
        return "No rows returned."

    display_df = df.copy()
    for column in display_df.select_dtypes(include=["float", "float64"]).columns:
        display_df[column] = display_df[column].map(lambda value: round(value, 2))

    return display_df.to_string(index=False)


def run_safe_query(sql: str, conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Validate, execute, and format a query in one safe wrapper."""

    # Never execute model output directly; validation must happen first.
    passed, reason = run_guardrails(sql, conn)
    if not passed:
        return {"success": False, "sql": sql, "results": "", "error": reason}

    try:
        df = execute_query(sql, conn)
    except duckdb.Error as exc:
        return {
            "success": False,
            "sql": sql,
            "results": "",
            "error": f"DuckDB execution failed: {exc}",
        }

    return {
        "success": True,
        "sql": sql,
        "results": format_results(df),
        "error": "",
    }
