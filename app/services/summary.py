from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db import postgres
from app.services.llm import weekly_summary
from app.services.timezones import get_tz

logger = logging.getLogger(__name__)


async def build_weekly_summary(user_id: int) -> str:
    """Сформировать сводку по заметкам пользователя за последние 7 дней."""
    time_from = datetime.now(timezone.utc) - timedelta(days=7)
    notes = await postgres.get_user_notes(user_id, time_from=time_from, limit=200)

    if not notes:
        return (
            "📭 За последнюю неделю заметок не было.\n\n"
            "Запиши пару мыслей — и в следующий раз я составлю обзор."
        )

    tz = await get_tz(user_id)
    payload = [
        {
            "created_at": n.created_at.astimezone(tz).strftime("%d.%m %H:%M"),
            "summary": n.summary,
            "full_text": n.full_text,
            "tags": n.tags,
        }
        for n in notes
    ]

    body = await weekly_summary(payload)
    return f"🗓 Сводка за неделю ({len(notes)} заметок)\n\n{body}"
