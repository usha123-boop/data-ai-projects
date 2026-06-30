"""CLI entrypoint for the Text-to-SQL with guardrails project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.execute import run_safe_query
from src.llm import generate_sql as llm_generate_sql
from src.llm import get_client
from src.setup_db import get_schema, setup_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "sales.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "output" / "sales.duckdb"


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the demo application."""

    parser = argparse.ArgumentParser(description="Text-to-SQL with validation guardrails")
    parser.add_argument("--setup", action="store_true", help="Load the CSV into DuckDB and print the schema")
    parser.add_argument("--query", type=str, help="Ask one natural-language question and run the full pipeline")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive query loop")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH), help="Path to the source CSV file")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to the DuckDB database file")
    return parser


def print_header(title: str) -> None:
    """Print a simple terminal section header."""

    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def answer_question(question: str, csv_path: str, db_path: str) -> None:
    """Run the full text-to-SQL pipeline for one question."""

    conn = setup_database(csv_path=csv_path, db_path=db_path)
    try:
        schema = get_schema(conn)
        client = get_client()

        print_header("Question")
        print(question)
        # The schema is fetched from DuckDB at runtime so the prompt always reflects reality.
        print("\nGenerating SQL with GPT-4o...")
        sql = llm_generate_sql(question=question, schema=schema, client=client)

        print_header("Generated SQL")
        print(sql)

        # This wrapper applies guardrails before running anything against DuckDB.
        response = run_safe_query(sql=sql, conn=conn)
        if not response["success"]:
            print_header("Guardrail / Execution Error")
            print(response["error"])
            return

        print_header("Results")
        print(response["results"])
    finally:
        conn.close()


def run_setup(csv_path: str, db_path: str) -> None:
    """Create the DuckDB table and show the detected schema."""

    conn = setup_database(csv_path=csv_path, db_path=db_path)
    try:
        print_header("Database Ready")
        print(f"CSV loaded from: {csv_path}")
        print(f"DuckDB file: {db_path}")
        print_header("Detected Schema")
        print(get_schema(conn))
    finally:
        conn.close()


def run_interactive(csv_path: str, db_path: str) -> None:
    """Open a simple REPL for repeated natural-language questions."""

    print("Starting interactive mode. Type 'exit' to stop.")
    while True:
        question = input("\nAsk a sales question> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            print("Please enter a question or type 'exit'.")
            continue
        try:
            answer_question(question=question, csv_path=csv_path, db_path=db_path)
        except Exception as exc:  # noqa: BLE001 - intentional for a user-facing CLI demo
            print_header("Pipeline Error")
            print(exc)


def main() -> None:
    """Parse CLI arguments and dispatch to the requested mode."""

    parser = build_parser()
    args = parser.parse_args()

    selected_modes = [args.setup, bool(args.query), args.interactive]
    if sum(selected_modes) != 1:
        parser.error("Choose exactly one mode: --setup, --query, or --interactive.")

    try:
        if args.setup:
            run_setup(csv_path=args.csv_path, db_path=args.db_path)
        elif args.query:
            answer_question(question=args.query, csv_path=args.csv_path, db_path=args.db_path)
        else:
            run_interactive(csv_path=args.csv_path, db_path=args.db_path)
    except Exception as exc:  # noqa: BLE001 - intentional for a user-facing CLI demo
        print_header("Fatal Error")
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
