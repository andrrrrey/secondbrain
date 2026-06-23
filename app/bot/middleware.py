from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message

import app.config.runtime as rt

logger = logging.getLogger(__name__)


class AccessMiddleware(BaseMiddleware):
    """Allow whitelisted user IDs, or everyone when open access is enabled."""

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if rt.open_access:
            return await handler(event, data)
        if rt.allowed_ids and event.from_user and event.from_user.id not in rt.allowed_ids:
            logger.warning("Unauthorized access attempt: user_id=%s", event.from_user.id)
            await event.answer("⛔ У вас нет доступа к этому боту.")
            return None
        return await handler(event, data)
