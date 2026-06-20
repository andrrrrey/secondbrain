from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

from app.config import settings
from app.models import Note, NoteType, Reminder, UserSettings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=settings.postgres_dsn, min_size=2, max_size=10)
        logger.info("PostgreSQL pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def init_tables() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id          TEXT PRIMARY KEY,
                user_id     BIGINT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                note_type   TEXT NOT NULL,
                full_text   TEXT NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                tags        JSONB NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes (user_id);
            CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes (created_at);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id         BIGINT PRIMARY KEY,
                timezone        TEXT NOT NULL DEFAULT 'Europe/Moscow',
                summary_enabled BOOLEAN NOT NULL DEFAULT true,
                summary_weekday INT NOT NULL DEFAULT 6,
                summary_hour    INT NOT NULL DEFAULT 18,
                last_summary_at TIMESTAMPTZ,
                reminder_lead_minutes INT NOT NULL DEFAULT 1440
            );

            ALTER TABLE user_settings
                ADD COLUMN IF NOT EXISTS reminder_lead_minutes INT NOT NULL DEFAULT 1440;

            CREATE TABLE IF NOT EXISTS reminders (
                id          TEXT PRIMARY KEY,
                user_id     BIGINT NOT NULL,
                note_id     TEXT,
                title       TEXT NOT NULL,
                event_at    TIMESTAMPTZ NOT NULL,
                remind_at   TIMESTAMPTZ NOT NULL,
                sent        BOOLEAN NOT NULL DEFAULT false,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (remind_at) WHERE sent = false;
            CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders (user_id);
        """)
        logger.info("PostgreSQL tables initialized")


async def save_note(note: Note) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO notes (id, user_id, created_at, note_type, full_text, summary, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            note.id,
            note.user_id,
            note.created_at,
            note.note_type.value,
            note.full_text,
            note.summary,
            json.dumps(note.tags, ensure_ascii=False),
        )


async def update_note(note_id: str, full_text: str, summary: str, tags: list[str]) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE notes SET full_text = $1, summary = $2, tags = $3
            WHERE id = $4
            """,
            full_text,
            summary,
            json.dumps(tags, ensure_ascii=False),
            note_id,
        )


async def delete_note(note_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM notes WHERE id = $1", note_id)


async def get_note_by_id(note_id: str) -> Note | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM notes WHERE id = $1", note_id)
    if row is None:
        return None
    return _row_to_note(row)


async def get_notes_by_ids(note_ids: list[str]) -> list[Note]:
    if not note_ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM notes WHERE id = ANY($1::text[])", note_ids
        )
    return [_row_to_note(r) for r in rows]


async def get_user_notes(
    user_id: int,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    note_type: str | None = None,
    tag: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Note]:
    pool = await get_pool()
    query = "SELECT * FROM notes WHERE user_id = $1"
    params: list = [user_id]
    idx = 2

    if time_from:
        query += f" AND created_at >= ${idx}"
        params.append(time_from)
        idx += 1
    if time_to:
        query += f" AND created_at <= ${idx}"
        params.append(time_to)
        idx += 1
    if note_type:
        query += f" AND note_type = ${idx}"
        params.append(note_type)
        idx += 1
    if tag:
        query += f" AND tags @> ${idx}::jsonb"
        params.append(json.dumps([tag], ensure_ascii=False))
        idx += 1

    query += f" ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1}"
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [_row_to_note(r) for r in rows]


async def count_user_notes(user_id: int) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count(*) as cnt FROM notes WHERE user_id = $1", user_id
        )
    return row["cnt"] if row else 0


async def get_user_tags(user_id: int) -> list[str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT jsonb_array_elements_text(tags) AS tag
            FROM notes WHERE user_id = $1
            ORDER BY tag
            """,
            user_id,
        )
    return [r["tag"] for r in rows]


# ── User settings ────────────────────────────────────────────────

async def get_user_settings(user_id: int) -> UserSettings | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM user_settings WHERE user_id = $1", user_id
        )
    if row is None:
        return None
    return _row_to_settings(row)


def _row_to_settings(row: asyncpg.Record) -> UserSettings:
    return UserSettings(
        user_id=row["user_id"],
        timezone=row["timezone"],
        summary_enabled=row["summary_enabled"],
        summary_weekday=row["summary_weekday"],
        summary_hour=row["summary_hour"],
        last_summary_at=row["last_summary_at"],
        reminder_lead_minutes=row["reminder_lead_minutes"],
    )


async def upsert_user_settings(s: UserSettings) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings
                (user_id, timezone, summary_enabled, summary_weekday, summary_hour,
                 last_summary_at, reminder_lead_minutes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id) DO UPDATE SET
                timezone = EXCLUDED.timezone,
                summary_enabled = EXCLUDED.summary_enabled,
                summary_weekday = EXCLUDED.summary_weekday,
                summary_hour = EXCLUDED.summary_hour,
                last_summary_at = EXCLUDED.last_summary_at,
                reminder_lead_minutes = EXCLUDED.reminder_lead_minutes
            """,
            s.user_id, s.timezone, s.summary_enabled,
            s.summary_weekday, s.summary_hour, s.last_summary_at,
            s.reminder_lead_minutes,
        )


async def set_user_timezone(user_id: int, tz: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, timezone)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET timezone = EXCLUDED.timezone
            """,
            user_id, tz,
        )


async def set_summary_schedule(
    user_id: int, weekday: int, hour: int, enabled: bool
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, summary_weekday, summary_hour, summary_enabled)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id) DO UPDATE SET
                summary_weekday = EXCLUDED.summary_weekday,
                summary_hour = EXCLUDED.summary_hour,
                summary_enabled = EXCLUDED.summary_enabled
            """,
            user_id, weekday, hour, enabled,
        )


async def set_reminder_lead(user_id: int, minutes: int) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, reminder_lead_minutes)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET
                reminder_lead_minutes = EXCLUDED.reminder_lead_minutes
            """,
            user_id, minutes,
        )


async def update_last_summary(user_id: int, dt: datetime) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE user_settings SET last_summary_at = $1 WHERE user_id = $2",
            dt, user_id,
        )


async def get_all_summary_users() -> list[UserSettings]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM user_settings WHERE summary_enabled = true"
        )
    return [_row_to_settings(r) for r in rows]


# ── Reminders ────────────────────────────────────────────────────

async def add_reminder(r: Reminder) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reminders
                (id, user_id, note_id, title, event_at, remind_at, sent, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            r.id, r.user_id, r.note_id, r.title,
            r.event_at, r.remind_at, r.sent, r.created_at,
        )


async def get_upcoming_reminders(user_id: int, limit: int = 20) -> list[Reminder]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM reminders
            WHERE user_id = $1 AND sent = false AND event_at >= now()
            ORDER BY event_at ASC
            LIMIT $2
            """,
            user_id, limit,
        )
    return [_row_to_reminder(r) for r in rows]


async def get_due_reminders(now: datetime) -> list[Reminder]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM reminders
            WHERE sent = false AND remind_at <= $1
            ORDER BY remind_at ASC
            """,
            now,
        )
    return [_row_to_reminder(r) for r in rows]


async def mark_reminder_sent(reminder_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE reminders SET sent = true WHERE id = $1", reminder_id
        )


async def get_reminder_by_id(reminder_id: str) -> Reminder | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM reminders WHERE id = $1", reminder_id)
    return _row_to_reminder(row) if row else None


async def delete_reminder(reminder_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM reminders WHERE id = $1", reminder_id)


async def delete_reminders_for_note(note_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM reminders WHERE note_id = $1", note_id)


def _row_to_reminder(row: asyncpg.Record) -> Reminder:
    return Reminder(
        id=row["id"],
        user_id=row["user_id"],
        note_id=row["note_id"],
        title=row["title"],
        event_at=row["event_at"],
        remind_at=row["remind_at"],
        sent=row["sent"],
        created_at=row["created_at"],
    )


def _row_to_note(row: asyncpg.Record) -> Note:
    tags = row["tags"]
    if isinstance(tags, str):
        tags = json.loads(tags)
    return Note(
        id=row["id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        note_type=NoteType(row["note_type"]),
        full_text=row["full_text"],
        summary=row["summary"],
        tags=tags,
    )
