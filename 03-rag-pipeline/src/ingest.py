"""Load support tickets and turn them into retrieval-friendly text chunks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {'ticket_id', 'subject', 'body', 'category', 'resolution'}


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Load the synthetic support ticket CSV and validate the schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f'Ticket file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f'Ticket file is missing required columns: {sorted(missing_columns)}'
        )
    return df


def prepare_chunks(df: pd.DataFrame) -> list[dict]:
    """Create one retrieval chunk per ticket.

    "Chunking" simply means deciding how much text to embed together. For this
    learning project we keep it intentionally simple: each support ticket
    becomes one chunk made from its subject plus body. That is a good teaching
    default because:

    1. each ticket is short enough to fit comfortably in one embedding request,
    2. keeping the full ticket together preserves meaning, and
    3. each retrieved chunk maps cleanly back to a real ticket ID.

    In larger production systems you might split long documents into smaller
    overlapping chunks so the retriever can focus on more specific sections.
    """
    chunks: list[dict] = []
    for row in df.itertuples(index=False):
        subject = str(row.subject).strip()
        body = str(row.body).strip()

        # Keeping labels like "Subject" and "Body" helps the model understand
        # the structure of the ticket when we later feed retrieved chunks back in.
        chunk_text = f"Subject: {subject}\nBody: {body}"
        chunks.append(
            {
                'ticket_id': str(row.ticket_id),
                'text': chunk_text,
                'category': str(row.category),
            }
        )
    return chunks
