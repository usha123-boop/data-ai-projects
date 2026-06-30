# de-ai-projects — AI for Data Engineers

> 5 hands-on projects that teach existing data engineers how to integrate AI into real data workflows.

No fluff. No toy examples. Each project mirrors something you'd actually build at work.

---

## Who is this for?

You're a **data engineer** who already knows:
- Python, SQL, PySpark
- Building pipelines, ETL/ELT
- Data modeling, data quality
- Tools like Airflow, dbt, Spark, Delta Lake

And you want to add **AI skills** to your stack — not just call an API, but genuinely understand how LLMs fit into data engineering workflows.

---

## Projects

| # | Project | Skill | Difficulty |
|---|---------|-------|------------|
| [01](./01-llm-enrichment-pipeline/) | LLM Enrichment Pipeline | Structured extraction from unstructured text | ✅ Beginner |
| [02](./02-airbnb-pipeline/) | Airbnb Pipeline | End-to-end pipeline with LLM enrichment layer | ✅ Easy |
| [03](./03-rag-pipeline/) | RAG Pipeline | Embeddings, vector search, retrieval-augmented generation | ⚠️ Medium |
| [04](./04-text-to-sql-guardrails/) | Text-to-SQL with Guardrails | NL → SQL generation + output validation | ⚠️ Medium-Hard |
| [05](./05-dimensional-modeler/) | LLM-Assisted Dimensional Modeler | LLM-driven schema design + fact/dim table building | ❌ Advanced |

---

## Stack

Every project uses the same portable, dependency-light stack:

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.9+ | Universal |
| LLM | OpenAI GPT-4o | Best quality, structured outputs |
| Embeddings | OpenAI text-embedding-3-small | Cheap, accurate |
| Storage | Parquet | Mirrors Delta Lake format |
| Analytics | DuckDB | Embedded SQL, no server, mirrors Spark SQL |
| Data | Synthetic CSVs (committed) | Run instantly, no downloads |

**Runs entirely on your laptop.** No Databricks account needed. No Docker. Just `pip install`.

---

## Adapting to Your Stack

Each project includes a section on adapting to production/cloud:

| Local | Databricks | Snowflake |
|-------|-----------|-----------|
| Pandas | PySpark (`mapInPandas`) | Snowpark |
| DuckDB | Delta Lake / Spark SQL | Snowflake SQL |
| Parquet files | Delta tables | Snowflake tables |
| OpenAI | Azure OpenAI / Bedrock | Same |

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/usha123=boop/data-ai-projects.git
cd de-ai-projects

# Start with project 01
cd 01-llm-enrichment-pipeline
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
python -m src.pipeline --input data/reviews.csv --output output/ --sample 10
```

**Start with `01-llm-enrichment-pipeline`** — it's the simplest and introduces the core pattern (LLM + structured output) that every other project builds on.

---

## Cost

All projects use OpenAI APIs. Running each project end-to-end on the sample data costs **< $0.10 total**. Use `--sample N` flags to limit rows and control spend.

---

## License

MIT
