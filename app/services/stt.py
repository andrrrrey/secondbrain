from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.services.llm import get_openai

logger = logging.getLogger(__name__)


async def transcribe(file_path: str | Path) -> str:
    """Transcribe audio file using Whisper API."""
    client = get_openai()
    path = Path(file_path)

    with open(path, "rb") as audio:
        resp = await client.audio.transcriptions.create(
            model=settings.whisper_model,
            file=audio,
            language="ru",
        )

    text = resp.text.strip()
    logger.info("Transcribed %s: %d chars", path.name, len(text))
    return text
