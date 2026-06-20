from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db import postgres, qdrant
from app.models import Note, NoteType, Reminder
from app.services.llm import extract_reminders, get_embedding, summarize_and_tag
from app.services.timezones import get_tz

logger = logging.getLogger(__name__)


async def _create_reminders_for_note(user_id: int, note_id: str, text: str) -> int:
    """Извлечь из текста события и сохранить напоминания (за сутки до события).

    Возвращает число созданных напоминаний. Ошибки не пробрасываются —
    они не должны мешать сохранению заметки."""
    try:
        tz = await get_tz(user_id)
        now_local = datetime.now(tz)
        now_utc = datetime.now(timezone.utc)
        user_settings = await postgres.get_user_settings(user_id)
        lead_minutes = user_settings.reminder_lead_minutes if user_settings else 1440
        events = await extract_reminders(text, now_local.isoformat(), str(tz))

        created = 0
        for ev in events:
            raw = ev.get("event_at")
            try:
                event_local = datetime.fromisoformat(raw)
            except (ValueError, TypeError):
                continue
            if event_local.tzinfo is None:
                event_local = event_local.replace(tzinfo=tz)
            event_utc = event_local.astimezone(timezone.utc)
            if event_utc <= now_utc:
                continue  # событие в прошлом — пропускаем
            remind_at = max(now_utc, event_utc - timedelta(minutes=lead_minutes))
            await postgres.add_reminder(Reminder(
                user_id=user_id,
                note_id=note_id,
                title=ev["title"],
                event_at=event_utc,
                remind_at=remind_at,
            ))
            created += 1
        if created:
            logger.info("Created %d reminder(s) for note %s", created, note_id)
        return created
    except Exception:
        logger.exception("Failed to extract reminders for note %s", note_id)
        return 0


async def process_and_save_note(
    user_id: int, text: str, note_type: NoteType
) -> tuple[Note, int]:
    """Process text: summarize, tag, embed, save to PG + Qdrant + извлечь напоминания.

    Возвращает (заметку, число созданных напоминаний)."""
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
    reminders = await _create_reminders_for_note(user_id, note.id, text)
    return note, reminders


async def update_existing_note(note: Note, new_text: str) -> tuple[Note, int]:
    """Update note text, re-summarize, re-tag, re-embed, пересоздать напоминания."""
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

    # Пересобираем напоминания: старые удаляем, извлекаем заново
    await postgres.delete_reminders_for_note(note.id)
    reminders = await _create_reminders_for_note(note.user_id, note.id, new_text)

    logger.info("Note updated: id=%s, tags=%s", note.id, note.tags)
    return note, reminders


async def delete_note_full(note_id: str) -> None:
    """Delete note from PG + Qdrant + связанные напоминания."""
    await postgres.delete_note(note_id)
    qdrant.delete_vector(note_id)
    await postgres.delete_reminders_for_note(note_id)
    logger.info("Note deleted: id=%s", note_id)
