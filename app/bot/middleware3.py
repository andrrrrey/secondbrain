from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message

import app.config.settings as rcfg

logger = logging.getLogger(__name__)


class AccessMiddleware(BaseMiddleware):
    """Allow only whitelisted user IDs (runtime-mutable)."""

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if rcfg.runtime_allowed_ids and event.from_user and event.from_user.id not in rcfg.runtime_allowed_ids:
            logger.warning("Unauthorized access attempt: user_id=%s", event.from_user.id)
            await event.answer("⛔ У вас нет доступа к этому боту.")
            return None
        return await handler(event, data)
