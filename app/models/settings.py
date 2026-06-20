from __future__ import annotations

import uuid
from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel, Field


# Python weekday(): Monday=0 .. Sunday=6
class Weekday(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class UserSettings(BaseModel):
    user_id: int
    timezone: str = "Europe/Moscow"
    summary_enabled: bool = True
    summary_weekday: int = 6  # воскресенье
    summary_hour: int = 18
    last_summary_at: datetime | None = None
    reminder_lead_minutes: int = 1440  # за сколько до события напоминать (24 ч)


class Reminder(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int
    note_id: str | None = None
    title: str
    event_at: datetime
    remind_at: datetime
    sent: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
