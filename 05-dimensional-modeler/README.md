# LLM-Assisted Dimensional Modeler

```text
              dim_customer        dim_product
              ┌──────────┐        ┌──────────┐
              │customer  │        │product   │
              │  _key PK │        │  _key PK │
              │name      │        │name      │
              │email     │        │category  │
              │city      │        │price     │
              └────┬─────┘        └────┬─────┘
                   │                   │
                   └────────┬──────────┘
                            │
                   ┌────────┴────────┐
                   │   fact_orders   │
                   │  order_key PK   │
                   │  customer_key FK│
                   │  product_key FK │
                   │  date_key FK    │
                   │  quantity       │
                   │  total_amount   │
                   └────────┬────────┘
                            │
                   ┌────────┴────────┐
                   │    dim_date     │
                   │  date_key PK    │
                   │  year/month/day │
                   │  quarter/week   │
                   └─────────────────┘
```

## What this project does

This project takes a flat, messy e-commerce orders extract, profiles the raw columns, asks an LLM to propose a star schema, and then uses Python to build the actual dimension and fact tables as Parquet files. You can inspect the suggested schema, review the generated tables, validate referential integrity, and query the result with DuckDB.

## What you'll learn

- How dimensional modeling turns raw operational data into analytics-friendly structures
- How to use an LLM to accelerate schema design
- The difference between fact tables and dimension tables
- Why star schemas are easier for BI and reporting workloads
- How surrogate keys simplify warehouse joins
- Why referential integrity checks matter in analytics engineering

## Why this matters for your resume

Most engineers can say they used SQL or Python. Far fewer can say they combined dimensional modeling with AI-assisted schema design. That blend of classic data warehousing plus practical LLM integration is rare, easy to demo in interviews, and highly relevant for modern analytics engineering roles.

## Prerequisites

- Python 3.10+
- Basic pandas familiarity
- Optional: an OpenAI API key for the live schema suggestion step

## Setup

1. Change into the project directory:
   ```bash
   cd /Users/Prateek/Development/GenAI/de-ai-projects/05-dimensional-modeler
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment file:
   ```bash
   cp .env.example .env
   ```
5. Add your API key to `.env` if you want the live LLM step.

## How to run

Run with the fallback schema (no API key required):

```bash
python -m src.pipeline --input data/orders.csv --output output/ --skip-llm
```

Run with GPT-4o schema suggestion:

```bash
python -m src.pipeline --input data/orders.csv --output output/
```

## Project structure

```text
05-dimensional-modeler/
├── data/
│   └── orders.csv              # 300-row flat operational-style orders extract
├── output/
│   ├── .gitkeep                # keeps the output directory in git
│   ├── dim_customer.parquet    # generated customer dimension
│   ├── dim_product.parquet     # generated product dimension
│   ├── dim_date.parquet        # generated date dimension
│   └── fact_orders.parquet     # generated fact table
├── src/
│   ├── __init__.py
│   ├── llm.py                  # OpenAI wrapper and schema suggestion call
│   ├── ingest.py               # raw CSV load + compact profiling summary
│   ├── suggest_schema.py       # LLM orchestration + readable schema display
│   ├── build_dims.py           # dimension table builders
│   ├── build_facts.py          # fact table builder with surrogate key joins
│   ├── validate_model.py       # referential integrity checks
│   └── pipeline.py             # end-to-end CLI pipeline
├── notebooks/
│   └── walkthrough.ipynb       # guided notebook walkthrough
├── .env.example
├── requirements.txt
└── README.md
```

## Sample output

### `dim_customer`

| customer_key | customer_id | customer_name   | customer_email             | customer_city | customer_country |
|---:|---|---|---|---|---|
| 1 | CUST-001 | Aanya Mehta   | aanya.mehta@example.com   | Mumbai        | India |
| 2 | CUST-002 | Benjamin Hall | benjamin.hall@example.com | Seattle       | USA |
| 3 | CUST-003 | Carla Gomez   | carla.gomez@example.com   | Madrid        | Spain |

### `fact_orders`

| order_key | customer_key | product_key | date_key | quantity | unit_price | discount_pct | total_amount | shipping_method | order_status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 12 | 7 | 20240103 | 2 | 79.99 | 0.10 | 143.98 | Express | Delivered |
| 2 | 4  | 18 | 20240105 | 1 | 24.50 | 0.00 | 24.50  | Standard | Processing |
| 3 | 9  | 3 | 20240106 | 4 | 15.00 | 0.05 | 57.00  | Same Day | Delivered |

## Estimated cost

The fallback mode is free. The live LLM mode typically uses a tiny prompt because only a profile and a small sample are sent, so the cost should usually stay well under a few cents per run depending on the model and current OpenAI pricing.

## Adapting to Databricks

- Replace local Parquet outputs with Delta Lake tables.
- Write dimensions and facts to Unity Catalog-managed schemas.
- Use Spark DataFrames instead of pandas for larger raw datasets.
- Keep the LLM suggestion step, but store approved schema JSON in a governance-controlled location.
- Add job orchestration with Databricks Workflows for repeatable production runs.

## Adapting the LLM

- Swap `gpt-4o` for another OpenAI model via `OPENAI_MODEL`.
- Replace the client in `src/llm.py` with Azure OpenAI, Anthropic, or an internal gateway.
- Add schema validation rules before accepting the model output.
- Cache approved schema proposals so repeated runs do not need another LLM call.
