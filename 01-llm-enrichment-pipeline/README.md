# LLM Enrichment Pipeline
Turn messy e-commerce review text into analytics-ready structured columns with GPT-4o.

```
reviews.csv → [Ingest] → [LLM Enrichment] → enriched.parquet
                         ↑
                   OpenAI GPT-4o
               (sentiment, category,
                issues, summary)
```

## What this project does

This project shows how data engineers can use an LLM as an enrichment step inside a familiar data pipeline. Instead of leaving customer reviews as raw text, the pipeline converts each review into structured fields like sentiment, issue category, summary, and a predicted rating. That makes the data much easier to group, filter, and load into dashboards or downstream warehouse tables.

The project is intentionally beginner-friendly. It uses pandas for ingestion, a thin OpenAI wrapper for model calls, and a simple orchestration module that writes Parquet output. If you already know ETL basics, this repo helps you see where LLM calls fit into real data workflows without adding extra framework complexity.

## What you'll learn

- How to load and validate raw CSV input before sending records to an LLM
- How to design a structured extraction prompt instead of asking for free-form text
- How to turn model responses into clean DataFrame columns
- How to handle row-level API failures without stopping the entire batch
- How to save enriched output as Parquet for analytics workflows
- How to adapt the same pattern to Databricks or other distributed data platforms

## Why this matters for your resume

This skill shows up in roles such as **Data Engineer**, **Analytics Engineer**, **Machine Learning Engineer**, **Applied AI Engineer**, and **Generative AI Engineer**. Employers increasingly want people who can enrich support tickets, product reviews, survey responses, or compliance notes with LLM-generated labels before loading the results into a warehouse.

Concrete use cases include customer feedback classification, help desk triage, review summarization, risk tagging, and turning messy text into dimensions that BI teams can actually use.

## Prerequisites

- Python 3.9+
- An OpenAI API key
- `pip`

## Setup

1. Clone the repository.
2. Change into this project directory:
   ```bash
   cd 01-llm-enrichment-pipeline
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
5. Open `.env` and add your `OPENAI_API_KEY`.

## How to run

### Option A: Notebook walkthrough

Start Jupyter from the project directory and open `notebooks/walkthrough.ipynb`:

```bash
jupyter notebook
```

### Option B: CLI pipeline

Run the full dataset:

```bash
python -m src.pipeline --input data/reviews.csv --output output/enriched.parquet
```

Run a smaller sample first:

```bash
python -m src.pipeline --input data/reviews.csv --output output/enriched-sample.parquet --sample 10
```

## Project structure

```
01-llm-enrichment-pipeline/
├── data/
│   └── reviews.csv              # 50 synthetic e-commerce reviews to enrich
├── output/
│   └── .gitkeep                 # Keeps the output folder in git
├── src/
│   ├── __init__.py              # Package marker for python -m execution
│   ├── llm.py                   # OpenAI wrapper and structured extraction helper
│   ├── ingest.py                # CSV loading and validation logic
│   ├── enrich.py                # Prompt building and row/batch enrichment
│   └── pipeline.py              # End-to-end CLI orchestration
├── notebooks/
│   └── walkthrough.ipynb        # Step-by-step interactive tutorial
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # Project guide
```

## Sample output

| review_id | product_name | sentiment | category | key_issues | summary | rating_prediction |
| --- | --- | --- | --- | --- | --- | --- |
| R003 | VoltGo USB-C Charger | negative | quality | ["stopped fast charging", "gets warm"] | Charger stopped fast charging quickly and appears unreliable. | 2 |
| R023 | VoltGo USB-C Charger | neutral | shipping | ["late delivery", "damaged box"] | Shipping was poor, but the charger itself still worked. | 3 |
| R039 | SnugPaws Pet Bed | positive | quality | ["soft fabric", "non-slip bottom"] | The pet bed feels comfortable and stable for an older dog. | 5 |

## Estimated cost

Running on all 50 sample rows costs approximately $0.05.

## Adapting to Databricks

The same enrichment pattern works in Spark by swapping pandas row loops for `mapInPandas`.

```python
from typing import Iterator
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
reviews_df = spark.read.csv("/dbfs/FileStore/reviews.csv", header=True)

def enrich_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    from src.llm import get_client
    from src.enrich import enrich_batch

    client = get_client()
    for pdf in iterator:
        yield enrich_batch(pdf, client=client)

enriched_df = reviews_df.mapInPandas(enrich_partition, schema="""
    review_id string,
    product_name string,
    review_text string,
    date string,
    sentiment string,
    category string,
    key_issues array<string>,
    summary string,
    rating_prediction int
""")
```

## Adapting the LLM

- **Azure OpenAI**: in `src/llm.py`, replace `OpenAI` with `AzureOpenAI`, then load `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, and API version values from `.env`.
- **Ollama**: replace the OpenAI client call with a request to your local Ollama endpoint, keep the prompt contract the same, and continue returning a Python `dict` so `enrich.py` does not need to change.
- **Amazon Bedrock**: wrap a Bedrock invocation in the same `extract_structured` interface so the rest of the pipeline stays provider-agnostic.
