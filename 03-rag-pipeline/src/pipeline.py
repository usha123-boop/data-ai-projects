"""Command-line entry point for building and querying the RAG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embed import embed_chunks, load_embeddings
from .ingest import load_tickets, prepare_chunks
from .llm import get_client
from .query import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'support_tickets.csv'
EMBEDDINGS_PATH = PROJECT_ROOT / 'output' / 'embeddings.parquet'


def build_index() -> None:
    """Run the indexing pipeline once and save embeddings to disk."""
    # Indexing is the offline step: read source data, chunk it, embed it, and
    # persist the vectors so online question answering can stay fast.
    print('Loading support tickets...')
    tickets_df = load_tickets(DATA_PATH)
    print(f'Loaded {len(tickets_df)} tickets from {DATA_PATH}')

    print('Preparing retrieval chunks...')
    chunks = prepare_chunks(tickets_df)
    print(f'Prepared {len(chunks)} chunks. Creating embeddings next...')

    client = get_client()
    embed_chunks(chunks, client, EMBEDDINGS_PATH)
    print('Index build complete.')


def run_query(question: str, k: int = 3) -> None:
    """Load the saved embeddings and answer a single question."""
    # Query-time work is intentionally light: load vectors, embed one question,
    # retrieve the best matches, then ask the LLM to answer from that evidence.
    print(f'Loading embeddings from {EMBEDDINGS_PATH}...')
    embeddings_df = load_embeddings(EMBEDDINGS_PATH)
    client = get_client()
    result = answer_question(question, embeddings_df, client, k=k)

    print("\nAnswer")
    print('------')
    print(result['answer'])
    print("\nSources")
    print('-------')
    for source in result['sources']:
        print(f'- {source}')


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description='Simple RAG pipeline over support tickets.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--build',
        action='store_true',
        help='Load tickets, generate embeddings, and write output/embeddings.parquet.',
    )
    group.add_argument(
        '--query',
        type=str,
        help='Ask a question against the indexed support tickets.',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='How many chunks to retrieve before asking the LLM for an answer.',
    )
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to build/query modes."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.build:
            build_index()
        else:
            run_query(args.query, k=args.k)
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
