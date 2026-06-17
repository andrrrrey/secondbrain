from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db import postgres, qdrant
from app.models import SearchResult
from app.services.llm import (
    answer_followup,
    ask_ai_with_context,
    classify_followup,
    generate_answer,
    get_embedding,
    parse_time_filter,
)

logger = logging.getLogger(__name__)


async def search_and_answer(user_id: int, query: str) -> str:
    """Full search pipeline: parse time → embed query → vector search → LLM answer (strict)."""
    context = await _get_relevant_context(user_id, query)

    if not context:
        return "В ваших заметках нет данных по этому вопросу."

    answer = await generate_answer(query, context)
    return answer


async def ask_ai(user_id: int, query: str, last_bot_message: str | None = None) -> str:
    """Smart AI pipeline: detect if followup → use last message context,
    otherwise search notes and answer freely."""

    # If there's a previous bot message, check if this is a follow-up
    if last_bot_message:
        is_followup = await classify_followup(query, last_bot_message)
        logger.info("Followup classification: query='%s', is_followup=%s", query, is_followup)

        if is_followup:
            return await answer_followup(query, last_bot_message)

    # Not a follow-up — do full RAG search
    context = await _get_relevant_context(user_id, query)
    answer = await ask_ai_with_context(query, context)
    return answer


async def _get_relevant_context(user_id: int, query: str) -> list[dict]:
    """Shared: extract time, embed, search, return context dicts.
    Falls back to search without time filter if time-filtered search finds nothing."""
    time_data = await parse_time_filter(query)
    time_from = _parse_iso(time_data.get("time_from"))
    time_to = _parse_iso(time_data.get("time_to"))

    logger.info("Search: query='%s', time_from=%s, time_to=%s", query, time_from, time_to)

    query_vector = await get_embedding(query)

    # First try with time filter
    similar = qdrant.search_similar(
        query_vector=query_vector,
        user_id=user_id,
        time_from=time_from,
        time_to=time_to,
        limit=5,
    )

    # Fallback: if time filter gave nothing but we had a filter — retry without it
    if not similar and (time_from or time_to):
        logger.info("Time-filtered search empty, retrying without time filter")
        similar = qdrant.search_similar(
            query_vector=query_vector,
            user_id=user_id,
            limit=5,
        )

    if not similar:
        return []

    note_ids = [nid for nid, _ in similar]
    notes = await postgres.get_notes_by_ids(note_ids)

    score_map = {nid: score for nid, score in similar}
    results = [
        SearchResult(note=n, score=score_map.get(n.id, 0.0))
        for n in notes
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    return [
        {
            "full_text": r.note.full_text,
            "created_at": r.note.created_at.astimezone(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M") if r.note.created_at.tzinfo else (r.note.created_at + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
            "tags": r.note.tags,
        }
        for r in results
    ]


def _parse_iso(value: str | None) -> datetime | None:
    if not value or value == "null":
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
