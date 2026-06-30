"""Helpers for talking to OpenAI models used by the RAG pipeline."""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client after loading environment variables.

    We keep the setup in one place so the rest of the project can simply ask
    for a ready-to-use client instead of repeating API key logic everywhere.
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError(
            'OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.'
        )
    return OpenAI(api_key=api_key)


def get_embedding(text: str, client: OpenAI) -> List[float]:
    """Convert text into an embedding vector using OpenAI.

    An embedding is a long list of numbers that captures the *meaning* of a
    piece of text. Texts that mean similar things tend to produce vectors that
    point in similar directions. That lets us compare a question like
    "How do I reset my password?" to stored ticket text and find the most
    relevant examples, even when the wording is not identical.

    We use ``text-embedding-3-small`` because it is a strong default learning
    model: inexpensive, fast, and good enough for many production use cases.
    Its vectors are 1,536 numbers long, which is a common size for modern
    embedding models—large enough to capture nuance, but still practical to
    store in a plain Parquet file.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError('Cannot create an embedding for empty text.')

    # The API returns a vector of floats. We keep it as a normal Python list here
    # so it is easy to serialize later before converting to NumPy for math.
    try:
        response = client.embeddings.create(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
            input=cleaned_text,
        )
    except Exception as exc:  # pragma: no cover - network/API error path
        raise RuntimeError(f'Embedding request failed: {exc}') from exc

    return list(response.data[0].embedding)


def generate_answer(question: str, context_chunks: list[str], client: OpenAI) -> str:
    """Ask GPT-4o to answer using only retrieved context.

    Retrieval-Augmented Generation (RAG) works by giving the language model a
    small, relevant set of documents right before we ask the final question.
    This is important because a general-purpose LLM may otherwise "fill in the
    blanks" with a plausible-sounding answer that is not grounded in our own
    support data. By explicitly telling the model to answer *only* from the
    retrieved ticket text, we reduce hallucinations and make the answer easier
    to audit.
    """
    if not context_chunks:
        return 'I could not find any relevant support tickets to answer that question.'

    numbered_context = "\n\n".join(
        f"Context chunk {index}:\n{chunk}"
        for index, chunk in enumerate(context_chunks, start=1)
    )

    # The system prompt sets strict behavior. In production RAG systems, this is
    # one of the simplest ways to tell the model to stay grounded in evidence.
    system_prompt = (
        'You are a careful customer support analyst. Answer ONLY using the '
        'provided context. If the context does not contain enough information, '
        'say that clearly instead of guessing.'
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{numbered_context}\n\n"
        'Write a concise answer grounded in the retrieved tickets. '
        'Do not invent policies, steps, or facts that are missing from the context.'
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o'),
            temperature=0,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
    except Exception as exc:  # pragma: no cover - network/API error path
        raise RuntimeError(f'Answer generation failed: {exc}') from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError('The model returned an empty answer.')
    return content.strip()
