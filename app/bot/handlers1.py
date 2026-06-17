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

# Temporary storage for transcribed texts awaiting user decision
# {user_id: transcribed_text}
_pending_voice: dict[int, str] = {}

COMMANDS = [
    BotCommand(command="start", description="Приветствие"),
    BotCommand(command="ask", description="Задать вопрос по заметкам"),
    BotCommand(command="save", description="Явно сохранить заметку"),
    BotCommand(command="help", description="Справка"),
]


async def set_bot_commands(bot: Bot) -> None:
    await bot.set_my_commands(COMMANDS)


def _voice_choice_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💾 Сохранить", callback_data="voice_save"),
            InlineKeyboardButton(text="🔍 Найти в заметках", callback_data="voice_search"),
        ]
    ])


# ── /start ──────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привет! Я твой AI Second Brain.\n\n"
        "📝 Текст → сохраняю как заметку\n"
        "🎤 Голосовое → транскрибирую и спрошу: сохранить или искать\n"
        "🔍 /ask <вопрос> — поиск по заметкам\n\n"
        "Примеры:\n"
        "/ask что я говорил про бота для спецификаций?\n"
        "/ask какие идеи я записывал?"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Как пользоваться:\n\n"
        "1️⃣ Пиши текст — сохранится как заметка\n"
        "2️⃣ Голосовое — транскрибирую, ты выберешь: сохранить или искать\n"
        "3️⃣ /ask <вопрос> — поиск по заметкам (текстом)\n"
        "4️⃣ /save <текст> — явно сохранить заметку\n"
    )


# ── /ask — search ──────────────────────────────────────────────

@router.message(Command("ask"))
async def cmd_ask(message: Message) -> None:
    query = (message.text or "").removeprefix("/ask").strip()
    if not query:
        await message.answer(
            "Напиши вопрос после /ask\n\n"
            "Пример: /ask что я говорил про архитектуру?"
        )
        return
    await _do_search(message, query)


# ── /save — explicit save ──────────────────────────────────────

@router.message(Command("save"))
async def cmd_save(message: Message) -> None:
    text = (message.text or "").removeprefix("/save").strip()
    if not text:
        await message.answer("Напиши текст после /save")
        return
    await _save_note(message, text, NoteType.TEXT)


# ── Voice → transcribe + ask user what to do ───────────────────

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

        _pending_voice[message.from_user.id] = text
        await processing_msg.edit_text(
            f"📝 Распознано:\n\n{text[:500]}\n\nЧто сделать?",
            reply_markup=_voice_choice_kb(),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Callbacks for voice choice ─────────────────────────────────

@router.callback_query(F.data == "voice_save")
async def cb_voice_save(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    text = _pending_voice.pop(user_id, None)

    if not text:
        await callback.answer("Текст не найден, отправь голосовое ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    try:
        note = await process_and_save_note(
            user_id=user_id, text=text, note_type=NoteType.VOICE,
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


@router.callback_query(F.data == "voice_search")
async def cb_voice_search(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    text = _pending_voice.pop(user_id, None)

    if not text:
        await callback.answer("Текст не найден, отправь голосовое ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("🔍 Ищу в заметках...")

    try:
        answer = await search_and_answer(user_id, text)
        await callback.message.answer(answer)
    except Exception:
        logger.exception("Search error")
        await callback.message.answer("⚠️ Ошибка при поиске.")


# ── Plain text → always save ───────────────────────────────────

@router.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_text(message: Message) -> None:
    text = message.text.strip()
    if not text:
        return
    await _save_note(message, text, NoteType.TEXT)


# ── Helpers ─────────────────────────────────────────────────────

async def _save_note(message: Message, text: str, note_type: NoteType) -> None:
    try:
        note = await process_and_save_note(
            user_id=message.from_user.id, text=text, note_type=note_type,
        )
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        await message.answer(
            f"✅ Заметка сохранена!\n\n"
            f"📋 {note.summary}\n"
            f"🏷 {tags_str}"
        )
    except Exception:
        logger.exception("Error saving note")
        await message.answer("⚠️ Ошибка при сохранении.")


async def _do_search(message: Message, query: str) -> None:
    await message.answer("🔍 Ищу в заметках...")
    try:
        answer = await search_and_answer(message.from_user.id, query)
        await message.answer(answer)
    except Exception:
        logger.exception("Search error")
        await message.answer("⚠️ Ошибка при поиске.")
