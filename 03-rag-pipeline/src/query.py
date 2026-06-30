"""End-to-end RAG query flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .llm import generate_answer, get_embedding
from .retrieve import retrieve_top_k


def answer_question(question: str, embeddings_df: pd.DataFrame, client, k: int = 3) -> dict:
    """Answer a question with retrieval-augmented generation.

    This is the full RAG loop in one function:

    1. Embed the user's question so it becomes a vector in the same semantic
       space as our stored ticket embeddings.
    2. Compare that question vector against every stored ticket vector.
    3. Retrieve the top-k most similar ticket chunks.
    4. Pass only those retrieved chunks to the LLM.
    5. Ask the LLM to answer using that context instead of guessing from
       general world knowledge.

    The important pattern is that retrieval narrows the evidence first, and
    generation happens second. That is the core idea behind most practical
    RAG systems in production.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError('Question cannot be empty.')

    # First embed the user's question so it lives in the same vector space as
    # the precomputed ticket embeddings we stored during indexing.
    question_embedding = np.array(get_embedding(cleaned_question, client), dtype=float)

    # Retrieval narrows the evidence to the most relevant ticket chunks.
    top_chunks = retrieve_top_k(question_embedding, embeddings_df, k=k)

    # Only the retrieved chunks are sent to the LLM. That grounding step is the
    # core reason RAG is more trustworthy than a plain free-form LLM call.
    context_chunks = [chunk['text'] for chunk in top_chunks]
    answer = generate_answer(cleaned_question, context_chunks, client)

    return {
        'question': cleaned_question,
        'answer': answer,
        'sources': [chunk['ticket_id'] for chunk in top_chunks],
    }
