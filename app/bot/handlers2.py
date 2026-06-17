from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import aiofiles
import aiohttp
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.models import NoteType
from app.services.notes import process_and_save_note
from app.services.search import search_and_answer
from app.services.stt import transcribe

logger = logging.getLogger(__name__)
router = Router()

# {user_id: {"text": str, "note_type": NoteType}}
_pending: dict[int, dict] = {}

COMMANDS = [
    BotCommand(command="start", description="Приветствие"),
    BotCommand(command="help", description="Справка"),
]


async def set_bot_commands(bot: Bot) -> None:
    await bot.set_my_commands(COMMANDS)


def _choice_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💾 Сохранить", callback_data="do_save"),
            InlineKeyboardButton(text="🔍 Найти в заметках", callback_data="do_search"),
        ]
    ])


# ── /start ──────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привет! Я твой AI Second Brain.\n\n"
        "📝 Отправь текст или голосовое — я спрошу что сделать:\n"
        "• Сохранить как заметку\n"
        "• Найти ответ среди заметок\n\n"
        "Просто пиши или говори — дальше разберёмся!"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Как пользоваться:\n\n"
        "1️⃣ Отправь текст или голосовое\n"
        "2️⃣ Выбери: сохранить или искать\n"
        "3️⃣ Готово!\n"
    )


# ── Voice ───────────────────────────────────────────────────────

@router.message(lambda m: m.voice is not None)
async def handle_voice(message: Message, bot: Bot) -> None:
    processing_msg = await message.answer("🎤 Обрабатываю голосовое...")

    voice = message.voice
    file = await bot.get_file(voice.file_id)
    file_url = f"https://api.telegram.org/file/bot{bot.token}/{file.file_path}"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(await resp.read())

    try:
        text = await transcribe(tmp_path)
        if not text:
            await processing_msg.edit_text("⚠️ Не удалось распознать речь.")
            return

        _pending[message.from_user.id] = {"text": text, "note_type": NoteType.VOICE}
        await processing_msg.edit_text(
            f"📝 Распознано:\n\n{text[:500]}\n\nЧто сделать?",
            reply_markup=_choice_kb(),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Plain text ──────────────────────────────────────────────────

@router.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_text(message: Message) -> None:
    text = message.text.strip()
    if not text:
        return

    _pending[message.from_user.id] = {"text": text, "note_type": NoteType.TEXT}
    await message.answer(
        f"Что сделать с этим сообщением?",
        reply_markup=_choice_kb(),
    )


# ── Callbacks ───────────────────────────────────────────────────

@router.callback_query(F.data == "do_save")
async def cb_save(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    data = _pending.pop(user_id, None)

    if not data:
        await callback.answer("Текст не найден, отправь сообщение ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    try:
        note = await process_and_save_note(
            user_id=user_id,
            text=data["text"],
            note_type=data["note_type"],
        )
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        await callback.message.answer(
            f"✅ Заметка сохранена!\n\n"
            f"📋 {note.summary}\n"
            f"🏷 {tags_str}"
        )
    except Exception:
        logger.exception("Error saving note")
        await callback.message.answer("⚠️ Ошибка при сохранении.")


@router.callback_query(F.data == "do_search")
async def cb_search(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    data = _pending.pop(user_id, None)

    if not data:
        await callback.answer("Текст не найден, отправь сообщение ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("🔍 Ищу в заметках...")

    try:
        answer = await search_and_answer(user_id, data["text"])
        await callback.message.answer(answer)
    except Exception:
        logger.exception("Search error")
        await callback.message.answer("⚠️ Ошибка при поиске.")
