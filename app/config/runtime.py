"""Mutable runtime state. Import this module and access variables directly."""

from app.config.settings import settings

# Allowed user IDs (loaded from .env, modifiable at runtime)
allowed_ids: set[int] = set(settings.initial_allowed_ids)
if settings.admin_user_id:
    allowed_ids.add(settings.admin_user_id)

# Admin user IDs (modifiable at runtime). Главный админ из .env снять нельзя.
admin_ids: set[int] = set()
if settings.admin_user_id:
    admin_ids.add(settings.admin_user_id)

# Current OpenAI API key (modifiable at runtime) — эмбеддинги, Whisper, vision
api_key: str = settings.openai_api_key

# RouterAI API key (modifiable at runtime) — Claude для обработки текста
routerai_key: str = settings.routerai_api_key
