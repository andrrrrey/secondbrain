from __future__ import annotations

import logging

from app.db import postgres, qdrant
from app.models import Note, NoteType
from app.services.llm import get_embedding, summarize_and_tag

logger = logging.getLogger(__name__)


async def process_and_save_note(user_id: int, text: str, note_type: NoteType) -> Note:
    """Process text: summarize, tag, embed, save to PG + Qdrant."""
    summary, tags = await summarize_and_tag(text)

    note = Note(
        user_id=user_id,
        note_type=note_type,
        full_text=text,
        summary=summary,
        tags=tags,
    )

    embedding = await get_embedding(text)

    await postgres.save_note(note)

    qdrant.upsert_vector(
        note_id=note.id,
        user_id=note.user_id,
        created_at=note.created_at,
        vector=embedding,
    )

    logger.info("Note saved: id=%s, type=%s, tags=%s", note.id, note.note_type, note.tags)
    return note


async def update_existing_note(note: Note, new_text: str) -> Note:
    """Update note text, re-summarize, re-tag, re-embed."""
    summary, tags = await summarize_and_tag(new_text)
    embedding = await get_embedding(new_text)

    await postgres.update_note(note.id, new_text, summary, tags)

    qdrant.upsert_vector(
        note_id=note.id,
        user_id=note.user_id,
        created_at=note.created_at,
        vector=embedding,
    )

    note.full_text = new_text
    note.summary = summary
    note.tags = tags

    logger.info("Note updated: id=%s, tags=%s", note.id, note.tags)
    return note


async def delete_note_full(note_id: str) -> None:
    """Delete note from PG + Qdrant."""
    await postgres.delete_note(note_id)
    qdrant.delete_vector(note_id)
    logger.info("Note deleted: id=%s", note_id)
