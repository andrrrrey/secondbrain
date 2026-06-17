from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class NoteType(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"


class Note(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int
    created_at: datetime = Field(default_factory=datetime.now)
    note_type: NoteType
    full_text: str
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    note: Note
    score: float


class SearchQuery(BaseModel):
    user_id: int
    query_text: str
    time_from: datetime | None = None
    time_to: datetime | None = None
