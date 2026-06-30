# RAG Pipeline (Retrieval-Augmented Generation)

```text
         ┌─────────────────────────────────────────┐
         │           INDEXING (run once)            │
         │  tickets.csv → chunk → embed → Parquet  │
         └─────────────────────────────────────────┘

         ┌─────────────────────────────────────────┐
         │           QUERYING (run anytime)         │
         │                                          │
         │  Question → embed → cosine similarity   │
         │      → top-3 chunks → GPT-4o → Answer  │
         └─────────────────────────────────────────┘
```

## What this project does

This project builds a full Retrieval-Augmented Generation (RAG) workflow over a synthetic support ticket dataset. You will index support tickets by converting them into embeddings, store those embeddings in a plain Parquet file, retrieve the most relevant tickets with NumPy-based cosine similarity, and then ask GPT-4o to answer questions using only the retrieved context.

## What you'll learn

- What embeddings are and why they let us compare text by meaning instead of exact keywords
- How vector search works with cosine similarity
- The end-to-end RAG pattern: index → retrieve → generate
- Why grounded generation reduces hallucinations and improves trust
- How to build a useful AI application without LangChain or a vector database

## Why this matters for your resume

RAG is one of the most common real-world AI application patterns. Building this project gives you a concrete example you can talk about for roles like RAG engineer, AI data engineer, analytics engineer working with LLM apps, or platform engineer supporting internal knowledge assistants.

## Prerequisites

- Python 3.10+
- An OpenAI API key
- Basic familiarity with pandas and the command line

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and add your API key:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set `OPENAI_API_KEY`.

## How to run

1. Build the index (runs once unless your data changes):

   ```bash
   python -m src.pipeline --build
   ```

2. Ask a grounded question:

   ```bash
   python -m src.pipeline --query "How do I reset my password?"
   ```

## Project structure

```text
03-rag-pipeline/
├── data/
│   └── support_tickets.csv      # 80 synthetic tickets used as the knowledge base
├── notebooks/
│   └── walkthrough.ipynb        # step-by-step notebook for learning the pipeline
├── output/
│   ├── .gitkeep                 # keeps the folder in git
│   └── embeddings.parquet       # generated at runtime, ignored by git
├── src/
│   ├── __init__.py              # package marker
│   ├── llm.py                   # OpenAI client, embeddings, and grounded answer generation
│   ├── ingest.py                # CSV loading and chunk preparation
│   ├── embed.py                 # embedding generation and Parquet persistence
│   ├── retrieve.py              # cosine similarity and top-k retrieval
│   ├── query.py                 # end-to-end RAG query flow
│   └── pipeline.py              # CLI entry point for build and query modes
├── .env.example                 # environment variable template
├── .gitignore                   # prevents committing runtime artifacts
├── requirements.txt             # Python dependencies
└── README.md                    # project guide
```

## Sample output

```text
$ python -m src.pipeline --query "How do I reset my password?"

Answer
------
Several tickets show that password reset issues are usually handled by sending a fresh reset email, invalidating expired reset links, and checking spam folders or cached browser sessions if the reset flow loops.

Sources
-------
- TEC-PWD-01
- TEC-PWD-02
- TEC-PWD-03
```

## Estimated cost

Embeddings are very cheap. `text-embedding-3-small` is roughly on the order of **$0.02 per 1M tokens**, so indexing a small support ticket dataset like this costs very little. The chat completion step is usually the more noticeable cost, but still modest for short grounded answers.

## Adapting to Databricks

The same pattern works well on Databricks:

- store chunked documents and embeddings in Delta Lake instead of Parquet,
- schedule indexing jobs with Databricks Workflows,
- replace the manual NumPy retrieval step with Databricks Vector Search when scale grows,
- add Unity Catalog governance around the source documents and generated outputs.

## Adapting the LLM

You can swap the LLM layer without rewriting the whole project. The main requirements are:

- one embedding model that turns text into vectors,
- one generation model that can answer from retrieved context,
- consistent prompt instructions that force grounded answers.

The same retrieval layer could be reused with Azure OpenAI, Anthropic (with an embedding provider), open-weight local models, or a production gateway that standardizes model access.
