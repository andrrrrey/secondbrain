from __future__ import annotations

import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.db import postgres

DEFAULT_TZ = "Europe/Moscow"

# Частые часовые пояса для кнопок онбординга (IANA, подпись)
COMMON_TIMEZONES: list[tuple[str, str]] = [
    ("Europe/Kaliningrad", "Калининград (UTC+2)"),
    ("Europe/Moscow", "Москва (UTC+3)"),
    ("Europe/Samara", "Самара (UTC+4)"),
    ("Asia/Yekaterinburg", "Екатеринбург (UTC+5)"),
    ("Asia/Omsk", "Омск (UTC+6)"),
    ("Asia/Krasnoyarsk", "Красноярск (UTC+7)"),
    ("Asia/Irkutsk", "Иркутск (UTC+8)"),
    ("Asia/Yakutsk", "Якутск (UTC+9)"),
    ("Asia/Vladivostok", "Владивосток (UTC+10)"),
    ("Europe/Kyiv", "Киев (UTC+2/+3)"),
    ("Asia/Almaty", "Алматы (UTC+5)"),
    ("Asia/Tashkent", "Ташкент (UTC+5)"),
]

# Кэш ZoneInfo по user_id, чтобы не дёргать БД на каждое форматирование
_tz_cache: dict[int, ZoneInfo] = {}


def parse_offset(text: str) -> str | None:
    """Разобрать ввод смещения вида '+3', 'UTC+5', '-2' в IANA-имя 'Etc/GMT∓N'.

    Внимание: в Etc/GMT знак инвертирован (Etc/GMT-3 == UTC+3)."""
    m = re.fullmatch(r"\s*(?:UTC|GMT|МСК)?\s*([+-]?)(\d{1,2})\s*", text, re.IGNORECASE)
    if not m:
        return None
    sign = m.group(1)
    hours = int(m.group(2))
    if hours > 14:
        return None
    # UTC+3 → Etc/GMT-3 ; UTC-5 → Etc/GMT+5
    if sign == "-":
        return f"Etc/GMT+{hours}"
    return f"Etc/GMT-{hours}"


def is_valid_tz(name: str) -> bool:
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


def get_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TZ)


async def get_tz(user_id: int) -> ZoneInfo:
    """Часовой пояс пользователя (с кэшем). Дефолт — Europe/Moscow."""
    cached = _tz_cache.get(user_id)
    if cached is not None:
        return cached
    s = await postgres.get_user_settings(user_id)
    tz_name = s.timezone if s else DEFAULT_TZ
    tz = get_zoneinfo(tz_name)
    _tz_cache[user_id] = tz
    return tz


def invalidate_tz_cache(user_id: int) -> None:
    _tz_cache.pop(user_id, None)
