"""Create and persist embeddings for ticket text chunks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .llm import get_embedding


def embed_chunks(chunks: list[dict], client, output_path: str | Path) -> pd.DataFrame:
    """Embed chunks and store them in Parquet.

    We precompute embeddings during an indexing step because embeddings are a
    reusable asset. If we embedded every ticket every time a question arrived,
    the system would be slow and unnecessarily expensive. Instead, we pay the
    embedding cost once, store the vectors, and later only embed the user's
    *question* at query time.

    The Parquet file stores four pieces of information for each chunk:
    - the ticket ID so we know the source,
    - the original text so we can show context to the LLM,
    - the category for simple metadata filtering or analysis,
    - the embedding vector itself serialized as JSON for portability.
    """
    if not chunks:
        raise ValueError('No chunks were provided for embedding.')

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failed: list[str] = []
    print(f'Generating embeddings for {len(chunks)} ticket chunks...')
    for chunk in tqdm(chunks, desc='Embedding tickets'):
        # We embed the full chunk text once during indexing so future queries can
        # reuse this vector instead of paying to re-embed the same ticket again.
        try:
            vector = get_embedding(chunk['text'], client)
        except Exception as exc:
            # A single failed API call (rate limit, timeout, etc.) should not
            # discard the embeddings already computed. We skip the bad chunk,
            # log it, and continue — the caller can decide whether to retry.
            print(f"\n⚠️  Skipping chunk {chunk['ticket_id']}: {exc}")
            failed.append(chunk['ticket_id'])
            continue
        records.append(
            {
                'ticket_id': chunk['ticket_id'],
                'text': chunk['text'],
                'category': chunk['category'],
                'embedding': json.dumps(vector),
            }
        )

    if not records:
        raise RuntimeError('All chunks failed to embed. No output written.')

    if failed:
        print(f"\n⚠️  {len(failed)} chunk(s) failed and were skipped: {failed}")

    embeddings_df = pd.DataFrame(records)
    embeddings_df.to_parquet(output_file, index=False)
    print(f'Saved embeddings to {output_file}')
    return embeddings_df


def load_embeddings(path: str | Path) -> pd.DataFrame:
    """Load embeddings from Parquet and convert JSON back to numpy arrays."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f'Embeddings file not found: {parquet_path}. Run the build step first.'
        )

    df = pd.read_parquet(parquet_path)
    if 'embedding' not in df.columns:
        raise ValueError('Embeddings parquet is missing the embedding column.')

    df = df.copy()
    # Parquet is great for tabular storage, but JSON keeps the list of floats easy
    # to inspect and portable across tools. We convert back to NumPy for math.
    df['embedding'] = df['embedding'].apply(
        lambda value: np.array(json.loads(value), dtype=float)
    )
    return df
