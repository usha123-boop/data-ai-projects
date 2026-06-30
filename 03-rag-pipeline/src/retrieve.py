"""Vector search using plain NumPy and cosine similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity compares the *direction* of two vectors rather than their
    raw size. In plain English:

    - the dot product tells us how much the vectors point in the same direction,
    - each vector magnitude tells us how long that vector is,
    - dividing by both magnitudes normalizes the score.

    That gives us a number between -1 and 1. For embeddings, higher values mean
    the two pieces of text are more semantically similar.
    """
    if a.shape != b.shape:
        raise ValueError('Vectors must have the same shape for cosine similarity.')

    # Dot product measures directional alignment; the norm terms normalize for
    # vector length so the score reflects similarity of meaning, not size alone.
    numerator = float(np.dot(a, b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def retrieve_top_k(
    query_embedding: np.ndarray, embeddings_df: pd.DataFrame, k: int = 3
) -> list[dict]:
    """Return the top-k most similar ticket chunks for a query embedding.

    Cosine similarity is usually a better fit than Euclidean distance for text
    embeddings because we mainly care about semantic *direction*. Two text
    vectors can have slightly different lengths for reasons that are not very
    meaningful, but if they point in a similar direction they often represent
    related topics. That is why cosine similarity is the standard baseline for
    simple vector search systems like this one.
    """
    if k <= 0:
        raise ValueError('k must be greater than 0.')
    if embeddings_df.empty:
        return []

    scored_rows: list[dict] = []
    for row in embeddings_df.itertuples(index=False):
        # This is the simplest possible vector search: compare the query vector to
        # every stored vector, score each one, then sort descending.
        score = cosine_similarity(query_embedding, row.embedding)
        scored_rows.append(
            {
                'ticket_id': row.ticket_id,
                'text': row.text,
                'category': row.category,
                'score': score,
            }
        )

    scored_rows.sort(key=lambda item: item['score'], reverse=True)
    return scored_rows[:k]
