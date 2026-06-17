from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import settings
import app.config.runtime as rt

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=rt.api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def reset_openai_client() -> None:
    """Reset client so it picks up a new API key."""
    global _client
    _client = None


async def get_embedding(text: str) -> list[float]:
    client = get_openai()
    resp = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return resp.data[0].embedding


async def summarize_and_tag(text: str) -> tuple[str, list[str]]:
    """Return (summary, tags) for a note text."""
    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — помощник для структурирования заметок. "
                    "Получив текст заметки, верни JSON: "
                    '{"summary": "краткое описание в 1-2 предложения", '
                    '"tags": ["тег1", "тег2", ...]}. '
                    "Теги — ключевые темы, на русском, 2-5 штук. "
                    "Отвечай ТОЛЬКО JSON, без markdown."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_completion_tokens=300,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        summary = data.get("summary", "")
        tags = data.get("tags", [])
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON: %s", raw)
        summary = ""
        tags = []
    return summary, tags


async def generate_answer(query: str, context_notes: list[dict]) -> str:
    """Generate answer based STRICTLY on found notes."""
    if not context_notes:
        return "В ваших заметках нет данных по этому вопросу."

    context_parts = []
    for n in context_notes:
        date_str = n.get("created_at", "?")
        text = n.get("full_text", "")
        context_parts.append(f"[{date_str}] {text}")

    context = "\n---\n".join(context_parts)

    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — ассистент, который отвечает на вопросы пользователя "
                    "СТРОГО на основе его заметок. "
                    "Если в заметках нет информации — скажи об этом. "
                    "Указывай даты заметок в ответе. "
                    "Не придумывай информацию, которой нет в заметках."
                ),
            },
            {
                "role": "user",
                "content": f"Мои заметки:\n{context}\n\nВопрос: {query}",
            },
        ],
        max_completion_tokens=1000,
    )
    return resp.choices[0].message.content or "Не удалось сформировать ответ."


async def ask_ai_with_context(
    query: str,
    context_notes: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """Ask AI freely, enriched with relevant notes as context.

    Unlike generate_answer, this function allows the AI to use
    both its general knowledge AND the user's notes to give
    a comprehensive answer. chat_history is a list of
    {"role": "user"/"assistant", "content": "..."} dicts.
    """
    context = ""
    if context_notes:
        context_parts = []
        for n in context_notes:
            date_str = n.get("created_at", "?")
            text = n.get("full_text", "")
            tags = ", ".join(n.get("tags", []))
            context_parts.append(f"[{date_str}] (теги: {tags})\n{text}")
        context = "\n---\n".join(context_parts)

    system_prompt = (
        "Ты — умный персональный ассистент пользователя. "
        "У тебя есть доступ к заметкам пользователя — используй их как контекст. "
        "Отвечай на вопрос, комбинируя свои знания и информацию из заметок. "
        "Если в заметках есть релевантная информация — ссылайся на неё с датами. "
        "Если заметок по теме нет — отвечай на основе своих знаний, "
        "но уточни, что в заметках пользователя информации по этой теме нет. "
        "Учитывай историю диалога, если она есть. "
        "Отвечай полезно, структурированно и по делу."
    )

    if context:
        user_content = (
            f"Мои заметки по теме:\n{context}\n\n"
            f"Мой запрос: {query}"
        )
    else:
        user_content = (
            f"В моих заметках ничего по теме не найдено.\n\n"
            f"Мой запрос: {query}"
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Add chat history (last N turns)
    if chat_history:
        messages.extend(chat_history[-10:])  # max 5 turns (10 messages)

    messages.append({"role": "user", "content": user_content})

    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.7,
        messages=messages,
        max_completion_tokens=1500,
    )
    result = resp.choices[0].message.content
    if not result or not result.strip():
        logger.warning("ask_ai empty response: %s", resp.choices[0])
        return "ИИ не смог сформировать ответ. Попробуй переформулировать вопрос."
    return result


async def parse_time_filter(query: str) -> dict:
    """Extract time references from user query."""
    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Из пользовательского запроса извлеки временные рамки, если они есть. "
                    "Верни JSON: "
                    '{"time_from": "ISO datetime или null", "time_to": "ISO datetime или null"}. '
                    "Если временных рамок нет — верни null для обоих полей. "
                    "Сегодняшняя дата подставляется автоматически. "
                    "Отвечай ТОЛЬКО JSON."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_completion_tokens=100,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def classify_followup(query: str, last_bot_message: str) -> bool:
    """Determine if user's query is a follow-up to the last bot message
    or a standalone new question."""
    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Определи, является ли сообщение пользователя продолжением диалога "
                    "(уточнение, вопрос про то же, о чём только что говорили) "
                    "или это новый отдельный вопрос/заметка.\n"
                    "Признаки продолжения: местоимения (это, он, она, там, туда), "
                    "краткие уточнения (а подробнее? расскажи больше, что это такое?), "
                    "ссылки на контекст предыдущего ответа.\n"
                    "Признаки нового вопроса: конкретная новая тема, нет связи с предыдущим.\n"
                    'Ответь ТОЛЬКО одним словом: "followup" или "new".'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Последний ответ бота:\n{last_bot_message[:1000]}\n\n"
                    f"Новое сообщение пользователя:\n{query}"
                ),
            },
        ],
        max_completion_tokens=10,
    )
    result = (resp.choices[0].message.content or "").strip().lower()
    return "followup" in result


async def answer_followup(query: str, last_bot_message: str) -> str:
    """Answer a follow-up question using the previous bot message as context."""
    client = get_openai()
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — умный персональный ассистент. "
                    "Пользователь задаёт уточняющий вопрос по предыдущему ответу. "
                    "Используй контекст предыдущего ответа и свои общие знания. "
                    "Отвечай полезно и по делу."
                ),
            },
            {"role": "assistant", "content": last_bot_message},
            {"role": "user", "content": query},
        ],
        max_completion_tokens=1500,
    )
    result = resp.choices[0].message.content
    if not result or not result.strip():
        return "ИИ не смог сформировать ответ. Попробуй переформулировать вопрос."
    return result
