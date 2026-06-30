"""Prompt-building helpers for text-to-SQL generation.

Prompt engineering matters a lot for SQL generation because SQL is extremely sensitive to small
mistakes. A model can understand the business intent but still fail if it invents a column,
forgets a grouping key, or returns prose around the query. The helpers here make the prompt
explicit and reusable so learners can inspect, tune, and extend it.
"""

from __future__ import annotations

import re


def build_system_prompt(schema: str) -> str:
    """Build the system prompt used for SQL generation.

    The prompt includes:
    - the schema so the model knows exactly what table and columns exist
    - a few-shot section with example question -> SQL pairs
    - strict rules that limit the output to safe, executable SQL

    Few-shot prompting works because we are giving the model concrete demonstrations of the
    transformation we want. When the examples resemble the real questions users ask, the model is
    more likely to match the desired style: valid column names, readable aliases, and a LIMIT.
    """

    # These examples teach the model the exact pattern we want it to imitate.
    examples = """
Example 1
Question: What are the top 5 products by revenue?
SQL: SELECT product, ROUND(SUM(total_amount), 2) AS revenue
FROM sales
GROUP BY product
ORDER BY revenue DESC
LIMIT 5

Example 2
Question: How many completed orders came from each region?
SQL: SELECT region, COUNT(*) AS completed_orders
FROM sales
WHERE status = 'completed'
GROUP BY region
ORDER BY completed_orders DESC
LIMIT 100

Example 3
Question: What is the average order value for each category?
SQL: SELECT category, ROUND(AVG(total_amount), 2) AS avg_order_value
FROM sales
GROUP BY category
ORDER BY avg_order_value DESC
LIMIT 100
""".strip()

    return f"""
You are a senior analytics engineer who writes safe DuckDB SQL.

You are working with exactly one table named sales.

Table schema:
{schema}

Follow these rules every time:
1. Return exactly one SQL query and nothing else.
2. Use only the sales table.
3. Generate read-only SQL only. Never write INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, COPY, or ATTACH.
4. Prefer simple, readable SQL that a data engineer can review.
5. Always include a LIMIT clause. Use LIMIT 100 unless the question explicitly asks for a smaller top N.
6. Use ROUND for currency aggregates when helpful.
7. Output SQL only. Do not wrap it in markdown backticks and do not add explanations.
8. If the question is ambiguous, make the most reasonable assumption using the available columns.

Few-shot examples:
{examples}
""".strip()


def extract_sql(llm_response: str) -> str:
    """Clean the model response and return raw SQL.

    Even when we ask for SQL only, models sometimes add code fences, labels like ``SQL:``, or a
    trailing semicolon. Cleaning these artifacts is a lightweight post-processing step that makes
    downstream validation more reliable. In production systems this kind of normalization is very
    common: prompts reduce errors, and deterministic code cleans up what still slips through.
    """

    cleaned = llm_response.strip()

    code_block_match = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    if cleaned.lower().startswith("sql:"):
        cleaned = cleaned[4:].strip()

    cleaned = cleaned.rstrip(";").strip()
    return cleaned
