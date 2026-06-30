# 04 - Text-to-SQL with Guardrails

## 1. Header
This learning project shows how to turn a natural-language analytics question into SQL with an LLM, validate that SQL with guardrails, and safely execute it against DuckDB.

## 2. ASCII flow diagram
```text
  Natural Language Question
           │
           ▼
  ┌─────────────────┐
  │  LLM (GPT-4o)   │  ← table schema + few-shot examples
  │  Generate SQL   │
  └────────┬────────┘
           │ raw SQL
           ▼
  ┌─────────────────┐
  │   Guardrails    │  ← SELECT only? LIMIT present? Valid syntax?
  │   Validate SQL  │
  └────────┬────────┘
           │ validated SQL
           ▼
  ┌─────────────────┐
  │     DuckDB      │
  │  Execute Query  │
  └────────┬────────┘
           │
           ▼
       Results ✅
```

## 3. What this project does
- Loads a synthetic e-commerce sales dataset into DuckDB.
- Builds a schema-aware prompt with few-shot examples.
- Uses GPT-4o to translate a business question into SQL.
- Validates the generated SQL before execution.
- Executes only safe SQL and formats the results for the terminal.
- Includes a notebook walkthrough so you can learn the pattern interactively.

## 4. What you'll learn
- How text-to-SQL systems work in practice.
- Why prompt engineering matters for SQL generation.
- How few-shot examples improve LLM output quality.
- Why guardrails are required before executing model output.
- How to validate LLM-generated SQL with deterministic code.

## 5. Why this matters for your resume
This project demonstrates a practical AI-for-data-engineering pattern: building an AI-powered data tool that is useful **and** safe. It is relevant for roles such as:
- Data engineer working on AI-enabled analytics experiences
- Analytics engineer building self-service BI copilots
- LLM / applied AI engineer building production data tools
- Platform engineer responsible for safe AI integrations

## 6. Prerequisites
- Python 3.10+
- Basic SQL knowledge
- An OpenAI API key
- Comfort running terminal commands

## 7. Setup
1. Move into the project directory:
   ```bash
   cd /Users/Prateek/Development/GenAI/de-ai-projects/04-text-to-sql-guardrails
   ```
2. Create and activate your virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create your environment file:
   ```bash
   cp .env.example .env
   ```
5. Add your `OPENAI_API_KEY` to `.env`.

## 8. How to run
### Inspect the database and schema
```bash
python3 -m src.pipeline --setup
```

### Ask a single question
```bash
python3 -m src.pipeline --query "What are the top 5 products by revenue?"
python3 -m src.pipeline --query "How many completed orders came from each region?"
python3 -m src.pipeline --query "What is the average order value for each category?"
```

### Start interactive mode
```bash
python3 -m src.pipeline --interactive
```

Type `exit` to stop the interactive loop.

## 9. Project structure (annotated)
```text
04-text-to-sql-guardrails/
├── data/
│   └── sales.csv              # synthetic 200-row e-commerce dataset
├── output/
│   ├── .gitkeep               # keeps the output directory in git
│   └── sales.duckdb           # created when you run the pipeline
├── src/
│   ├── __init__.py            # package marker
│   ├── llm.py                 # OpenAI client + SQL generation call
│   ├── setup_db.py            # load CSV into DuckDB, inspect schema
│   ├── generate_sql.py        # prompt builder + response cleanup
│   ├── validate.py            # SQL guardrails before execution
│   ├── execute.py             # safe query execution + formatting
│   └── pipeline.py            # CLI orchestration and interactive loop
├── notebooks/
│   └── walkthrough.ipynb      # step-by-step learning notebook
├── .env.example               # required environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # project guide
```

## 10. Sample output
Example question:
```text
What are the top 5 products by revenue?
```

Example generated SQL:
```sql
SELECT product, ROUND(SUM(total_amount), 2) AS revenue
FROM sales
GROUP BY product
ORDER BY revenue DESC
LIMIT 5
```

Example results:
```text
         product  revenue
   Laptop Pro 14 42607.20
    Smartphone X 31375.10
      4K Monitor 21585.90
Wireless Earbuds  7621.35
    Denim Jacket  4040.60
```

## 11. Estimated cost
This project is inexpensive to run because prompts are short and result sets are tiny. For most demo questions, the OpenAI cost should be well under **$0.01 per query** with GPT-4o, but always check the latest pricing before using it at scale.

## 12. Adapting to Databricks
To adapt this pattern to Databricks:
- Replace DuckDB with a Databricks SQL warehouse connection.
- Keep the same prompt-building approach, but provide the warehouse schema instead of DuckDB schema.
- Preserve the guardrails layer before execution.
- Add stronger controls such as table allow-lists, approved semantic metrics, and query auditing.
- Consider routing only validated SQL to a service principal with read-only permissions.

## 13. Adapting the LLM
You can swap GPT-4o for another model by changing the client setup in `src/llm.py` and updating `.env`.

Common adaptations:
- Azure OpenAI for enterprise deployments
- Anthropic or Gemini via a provider-specific SDK
- An internal gateway that standardizes model access
- Fine-tuned or domain-specialized models for warehouse-specific SQL styles

The architecture stays the same: **prompt -> generate SQL -> validate -> execute -> return results**.
