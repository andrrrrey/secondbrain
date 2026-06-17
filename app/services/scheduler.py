from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot

from app.db import postgres
from app.services.summary import build_weekly_summary
from app.services.timezones import get_zoneinfo

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60  # секунд


async def _send_due_reminders(bot: Bot) -> None:
    now = datetime.now(timezone.utc)
    due = await postgres.get_due_reminders(now)
    for r in due:
        try:
            s = await postgres.get_user_settings(r.user_id)
            tz = get_zoneinfo(s.timezone if s else "Europe/Moscow")
            when = r.event_at.astimezone(tz).strftime("%d.%m.%Y в %H:%M")
            await bot.send_message(
                r.user_id,
                f"⏰ Напоминание!\n\n📌 {r.title}\n🗓 {when}",
            )
            await postgres.mark_reminder_sent(r.id)
            logger.info("Reminder sent: id=%s user=%s", r.id, r.user_id)
        except Exception:
            logger.exception("Failed to send reminder %s", r.id)


async def _send_weekly_summaries(bot: Bot) -> None:
    users = await postgres.get_all_summary_users()
    now_utc = datetime.now(timezone.utc)
    for s in users:
        try:
            tz = get_zoneinfo(s.timezone)
            local = now_utc.astimezone(tz)
            if local.weekday() != s.summary_weekday or local.hour != s.summary_hour:
                continue
            # Защита от повторной отправки в тот же день
            if s.last_summary_at is not None:
                last_local = s.last_summary_at.astimezone(tz)
                if last_local.date() == local.date():
                    continue
            text = await build_weekly_summary(s.user_id)
            await bot.send_message(s.user_id, text)
            await postgres.update_last_summary(s.user_id, now_utc)
            logger.info("Weekly summary sent: user=%s", s.user_id)
        except Exception:
            logger.exception("Failed to send weekly summary to %s", s.user_id)


async def run_scheduler(bot: Bot) -> None:
    """Фоновый цикл: напоминания (за сутки) и еженедельные сводки."""
    logger.info("Scheduler started")
    while True:
        try:
            await _send_due_reminders(bot)
            await _send_weekly_summaries(bot)
        except Exception:
            logger.exception("Scheduler iteration failed")
        await asyncio.sleep(CHECK_INTERVAL)
