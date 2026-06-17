"""Mutable runtime state. Import this module and access variables directly."""

from app.config.settings import settings

# Allowed user IDs (loaded from .env, modifiable at runtime)
allowed_ids: set[int] = set(settings.initial_allowed_ids)
if settings.admin_user_id:
    allowed_ids.add(settings.admin_user_id)

# Current API key (modifiable at runtime)
api_key: str = settings.openai_api_key
