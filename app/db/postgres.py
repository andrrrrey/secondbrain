from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

from app.config import settings
from app.models import Note, NoteType

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
