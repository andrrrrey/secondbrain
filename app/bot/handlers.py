from __future__ import annotations

import logging
import tempfile
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiofiles
import aiohttp
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.db import postgres
from app.models import NoteType, UserSettings
from app.services.llm import analyze_image, classify_image_intent, reset_openai_client
from app.services.notes import delete_note_full, process_and_save_note, update_existing_note
from app.services.search import ask_ai, search_and_answer
from app.services.stt import transcribe
from app.services.summary import build_weekly_summary
from app.services import timezones
from app.config.settings import settings
import app.config.runtime as rt

logger = logging.getLogger(__name__)
router = Router()

NOTES_PER_PAGE = 5

WEEKDAY_NAMES = [
    "Понедельник", "Вторник", "Среда", "Четверг",
    "Пятница", "Суббота", "Воскресенье",
]

TYPE_ICONS = {
    NoteType.VOICE: "🎤",
    NoteType.IMAGE: "🖼",
    NoteType.TEXT: "📝",
}


def fmt_date(dt, tz: ZoneInfo, fmt: str = "%d.%m.%Y %H:%M") -> str:
    """Format datetime in the user's timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime(fmt)


# ── FSM States ──────────────────────────────────────────────────

class EditNote(StatesGroup):
    waiting_for_text = State()


class TzStates(StatesGroup):
    waiting_offset = State()


class AdminStates(StatesGroup):
    waiting_add_user = State()
    waiting_remove_user = State()
    waiting_api_key = State()


# ── Pending voice/text storage ──────────────────────────────────

_pending: dict[int, dict] = {}

# Last bot message per user for follow-up detection
_last_bot_message: dict[int, str] = {}

COMMANDS = [
    BotCommand(command="start", description="Главное меню"),
    BotCommand(command="help", description="Справка"),
    BotCommand(command="reminders", description="Мои напоминания"),
    BotCommand(command="summary", description="Сводка за неделю"),
    BotCommand(command="settings", description="Настройки (пояс, сводка)"),
    BotCommand(command="admin", description="Управление ботом (админ)"),
    BotCommand(command="api", description="Сменить API-ключ (админ)"),
]


async def set_bot_commands(bot: Bot) -> None:
    await bot.set_my_commands(COMMANDS)


# ═══════════════════════════════════════════════════════════════
#  KEYBOARDS
# ═══════════════════════════════════════════════════════════════

def kb_main_choice() -> InlineKeyboardMarkup:
    """After transcription/input: save, search, ask AI, or manage."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Запомнить", callback_data="do_save"),
            InlineKeyboardButton(text="🔍 Найти в памяти", callback_data="do_search"),
        ],
        [
            InlineKeyboardButton(text="🤖 Спросить у ИИ", callback_data="do_ask_ai"),
        ],
        [
            InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu"),
        ],
    ])


def kb_main_menu() -> InlineKeyboardMarkup:
    """Главное меню с доступом к заметкам, напоминаниям, сводке и настройкам."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu")],
        [
            InlineKeyboardButton(text="⏰ Напоминания", callback_data="reminders"),
            InlineKeyboardButton(text="🗓 Сводка за неделю", callback_data="do_summary"),
        ],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
    ])


def kb_after_save() -> InlineKeyboardMarkup:
    """After saving: edit tags, add more, search."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🏷 Изменить теги", callback_data="edit_tags_last"),
            InlineKeyboardButton(text="➕ Добавить ещё мысль", callback_data="add_more"),
        ],
        [
            InlineKeyboardButton(text="🔍 Найти в памяти", callback_data="do_search_fresh"),
        ],
    ])


def kb_after_search() -> InlineKeyboardMarkup:
    """After search/AI result."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💾 Сохранить ответ как заметку", callback_data="save_answer"),
        ],
        [
            InlineKeyboardButton(text="➕ Новая заметка", callback_data="add_more"),
            InlineKeyboardButton(text="🔍 Найти ещё", callback_data="do_search_fresh"),
        ],
        [
            InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu"),
        ],
    ])


def kb_manage_filters() -> InlineKeyboardMarkup:
    """Note management: filter options."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📅 По дате", callback_data="filter_date"),
            InlineKeyboardButton(text="🏷 По тегу", callback_data="filter_tag"),
        ],
        [
            InlineKeyboardButton(text="📝 По типу заметки", callback_data="filter_type"),
            InlineKeyboardButton(text="🔍 По ключевым словам", callback_data="filter_kw"),
        ],
        [
            InlineKeyboardButton(text="📋 Все заметки", callback_data="notes_all:0"),
        ],
        [
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="home"),
        ],
    ])


def kb_note_actions(note_id: str) -> InlineKeyboardMarkup:
    """Actions for a single note."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👁 Просмотреть", callback_data=f"note_view:{note_id}"),
            InlineKeyboardButton(text="✏️ Редактировать", callback_data=f"note_edit:{note_id}"),
        ],
        [
            InlineKeyboardButton(text="🗑 Удалить", callback_data=f"note_del:{note_id}"),
            InlineKeyboardButton(text="⬅️ Назад", callback_data="manage_menu"),
        ],
    ])


def kb_confirm_delete(note_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да, удалить", callback_data=f"note_del_yes:{note_id}"),
            InlineKeyboardButton(text="❌ Нет, отмена", callback_data=f"note_del_no:{note_id}"),
        ],
    ])


def kb_after_action() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu"),
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="home"),
        ],
    ])


def kb_home() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="home")],
    ])


def _notes_list_kb(notes, offset: int, tz: ZoneInfo, prefix: str = "notes_all") -> InlineKeyboardMarkup:
    """Build paginated notes list with note buttons."""
    buttons = []
    for n in notes:
        date_str = fmt_date(n.created_at, tz, "%d.%m.%Y")
        type_icon = TYPE_ICONS.get(n.note_type, "📝")
        label = f"{date_str} {type_icon} {n.summary[:40]}"
        buttons.append([InlineKeyboardButton(text=label, callback_data=f"note_actions:{n.id}")])

    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"{prefix}:{offset - NOTES_PER_PAGE}"))
    if len(notes) == NOTES_PER_PAGE:
        nav.append(InlineKeyboardButton(text="➡️ Вперёд", callback_data=f"{prefix}:{offset + NOTES_PER_PAGE}"))
    if nav:
        buttons.append(nav)

    buttons.append([InlineKeyboardButton(text="🏠 Главное меню", callback_data="home")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ═══════════════════════════════════════════════════════════════
#  /start & /help
# ═══════════════════════════════════════════════════════════════

GREETING = (
    "👋 Привет! Я — твой второй мозг.\n\n"
    "Вот что я умею:\n"
    "📝 Сохраняю заметки из текста, голоса и фото\n"
    "🔍 Ищу по смыслу в твоих заметках\n"
    "🤖 Отвечаю на вопросы с учётом заметок\n"
    "⏰ Сам нахожу даты и события в заметках и напоминаю за сутки\n"
    "🗓 Раз в неделю присылаю сводку о главном\n\n"
    "Просто отправь мне текст, голосовое или фото 👇"
)


def kb_timezone() -> InlineKeyboardMarkup:
    """Клавиатура выбора часового пояса."""
    rows = []
    row = []
    for iana, label in timezones.COMMON_TIMEZONES:
        row.append(InlineKeyboardButton(text=label, callback_data=f"tz:{iana}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="✍️ Ввести смещение вручную", callback_data="tz_manual")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    _last_bot_message.pop(user_id, None)

    settings_row = await postgres.get_user_settings(user_id)
    if settings_row is None:
        # Первый запуск — спрашиваем часовой пояс
        await message.answer(
            "👋 Привет! Прежде чем начать, выбери свой часовой пояс —\n"
            "чтобы напоминания и сводки приходили по твоему местному времени.",
            reply_markup=kb_timezone(),
        )
        return

    await message.answer(GREETING, reply_markup=kb_main_menu())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Как пользоваться\n\n"
        "💾 *Сохранить заметку*\n"
        "Отправь текст, голосовое или фото → нажми «✅ Запомнить».\n"
        "Пример: «Идея: сделать лендинг для проекта».\n\n"
        "🔍 *Найти в памяти*\n"
        "Отправь вопрос → «🔍 Найти в памяти».\n"
        "Пример: «что я записывал про лендинг?»\n\n"
        "🤖 *Спросить у ИИ*\n"
        "Отправь вопрос → «🤖 Спросить у ИИ» (ответит с учётом заметок).\n\n"
        "⏰ *Напоминания*\n"
        "Просто запиши мысль с датой — я напомню за сутки.\n"
        "Пример: «хочу в театр в среду в 19:00».\n"
        "Список и удаление — команда /reminders\n\n"
        "🗓 *Сводка за неделю*\n"
        "Команда /summary — обзор в любой момент.\n"
        "День и время авто-сводки — в /settings\n\n"
        "🌍 *Часовой пояс*\n"
        "Сменить — в /settings",
        parse_mode="Markdown",
    )


# ═══════════════════════════════════════════════════════════════
#  TIMEZONE onboarding
# ═══════════════════════════════════════════════════════════════

async def _save_tz_and_greet(user_id: int, tz_name: str, send) -> None:
    await postgres.set_user_timezone(user_id, tz_name)
    timezones.invalidate_tz_cache(user_id)
    await send(
        f"✅ Часовой пояс сохранён: {tz_name}\n\n{GREETING}",
        reply_markup=kb_main_menu(),
    )


@router.callback_query(F.data.startswith("tz:"))
async def cb_set_tz(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer()
    tz_name = callback.data.split(":", 1)[1]
    if not timezones.is_valid_tz(tz_name):
        await callback.message.answer("⚠️ Не удалось распознать пояс. Попробуй ещё раз.")
        return
    await _save_tz_and_greet(callback.from_user.id, tz_name, callback.message.answer)


@router.callback_query(F.data == "tz_manual")
async def cb_tz_manual(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.set_state(TzStates.waiting_offset)
    await callback.message.answer(
        "✍️ Отправь своё смещение от UTC, например: +3, -5, UTC+4"
    )


async def _finish_tz_offset(message: Message, state: FSMContext, text: str) -> None:
    await state.clear()
    tz_name = timezones.parse_offset(text)
    if not tz_name or not timezones.is_valid_tz(tz_name):
        await message.answer(
            "⚠️ Не понял смещение. Пример: +3 или -5. Попробуй ещё раз через /timezone."
        )
        return
    await _save_tz_and_greet(message.from_user.id, tz_name, message.answer)


@router.message(Command("timezone"))
async def cmd_timezone(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "🌍 Выбери часовой пояс:",
        reply_markup=kb_timezone(),
    )


# ═══════════════════════════════════════════════════════════════
#  HOME callback
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "home")
async def cb_home(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer()
    await callback.message.edit_text(GREETING, reply_markup=kb_main_menu())


# ═══════════════════════════════════════════════════════════════
#  INPUT: Voice → transcribe + choice
# ═══════════════════════════════════════════════════════════════

@router.message(lambda m: m.voice is not None)
async def handle_voice(message: Message, bot: Bot, state: FSMContext) -> None:
    # If we're editing a note, don't intercept
    current = await state.get_state()
    if current == EditNote.waiting_for_text.state:
        await message.answer("Сейчас жду новый текст заметки. Отправь текстом, пожалуйста.")
        return

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
            f"📝 Распознано:\n\n{text[:500]}\n\n"
            "Вы хотите запомнить это как заметку или хотите найти в памяти?",
            reply_markup=kb_main_choice(),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
#  INPUT: Photo → analyze + choice
# ═══════════════════════════════════════════════════════════════

@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, state: FSMContext) -> None:
    current = await state.get_state()
    if current == EditNote.waiting_for_text.state:
        await message.answer("Сейчас жду новый текст заметки. Отправь текстом, пожалуйста.")
        return

    processing_msg = await message.answer("🖼 Анализирую изображение...")

    photo = message.photo[-1]  # самое крупное
    file = await bot.get_file(photo.file_id)
    file_url = f"https://api.telegram.org/file/bot{bot.token}/{file.file_path}"

    caption = (message.caption or "").strip()
    user_id = message.from_user.id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                image_bytes = await resp.read()

        analysis = await analyze_image(image_bytes, mime="image/jpeg")
        if not analysis:
            await processing_msg.edit_text("⚠️ Не удалось распознать изображение.")
            return

        # Если есть подпись — учитываем её как команду; иначе данные = распознанное
        combined = f"{caption}\n\n📄 Распознано на изображении:\n{analysis}" if caption else analysis
        header = f"🖼 Вот что на изображении:\n\n{analysis[:1500]}"

        if caption:
            intent = await classify_image_intent(caption)
            logger.info("Image caption intent: %s (caption=%r)", intent, caption)

            if intent in ("note", "reminder"):
                _pending.pop(user_id, None)
                await processing_msg.edit_text(header)
                await _perform_save(user_id, combined, NoteType.IMAGE, processing_msg)
                return
            if intent == "ask":
                _pending.pop(user_id, None)
                await processing_msg.edit_text(header)
                await _perform_ask(user_id, combined, processing_msg)
                return
            if intent == "search":
                _pending.pop(user_id, None)
                await processing_msg.edit_text(header)
                await _perform_search(user_id, combined, processing_msg)
                return
            # intent == "unclear" → уточняем у пользователя

        # Нет подписи или намерение неясно — спрашиваем кнопками
        _pending[user_id] = {"text": combined, "note_type": NoteType.IMAGE}
        clarify = (
            "\n\nНе понял, что сделать с подписью. Выбери действие:"
            if caption else "\n\nЧто сделать с этими данными?"
        )
        await processing_msg.edit_text(header + clarify, reply_markup=kb_main_choice())
    except Exception:
        logger.exception("Image analysis error")
        await processing_msg.edit_text("⚠️ Ошибка при анализе изображения.")


# ═══════════════════════════════════════════════════════════════
#  INPUT: Text → choice (unless in FSM edit mode)
# ═══════════════════════════════════════════════════════════════

@router.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_text(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text:
        return

    current = await state.get_state()

    # If editing a note — handle FSM
    if current == EditNote.waiting_for_text.state:
        await _finish_edit(message, state, text)
        return

    # If entering timezone offset manually
    if current == TzStates.waiting_offset.state:
        await _finish_tz_offset(message, state, text)
        return

    # If adding user via admin
    if current == AdminStates.waiting_add_user.state:
        await _finish_add_user(message, state, text)
        return

    # If changing API key
    if current == AdminStates.waiting_api_key.state:
        await _finish_api_key(message, state, text)
        return

    _pending[message.from_user.id] = {"text": text, "note_type": NoteType.TEXT}
    await message.answer(
        "Вы хотите запомнить это как заметку или хотите найти в памяти?",
        reply_markup=kb_main_choice(),
    )


# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "do_save")
async def cb_save(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    data = _pending.pop(user_id, None)
    if not data:
        await callback.answer("Текст не найден, отправь сообщение ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await _perform_save(user_id, data["text"], data["note_type"], callback.message)


async def _perform_save(user_id: int, text: str, note_type: NoteType, message: Message) -> None:
    try:
        note, reminders = await process_and_save_note(
            user_id=user_id, text=text, note_type=note_type,
        )
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        rem_line = (
            f"\n⏰ Поставил напоминаний: {reminders} (напомню за сутки)"
            if reminders else ""
        )
        await message.answer(
            f"✅ Запомнил!\n\n"
            f"📋 {note.summary}\n"
            f"🏷 {tags_str}"
            f"{rem_line}",
            reply_markup=kb_after_save(),
        )
    except Exception:
        logger.exception("Error saving note")
        await message.answer("⚠️ Ошибка при сохранении.", reply_markup=kb_home())


# ═══════════════════════════════════════════════════════════════
#  SEARCH
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "do_search")
async def cb_search(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    data = _pending.pop(user_id, None)
    if not data:
        await callback.answer("Текст не найден, отправь сообщение ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await _perform_search(user_id, data["text"], callback.message)


async def _perform_search(user_id: int, text: str, message: Message) -> None:
    await message.answer("🔍 Ищу в памяти...")
    try:
        answer = await search_and_answer(user_id, text)
        _last_bot_message[user_id] = answer
        await message.answer(answer, reply_markup=kb_after_search())
    except Exception:
        logger.exception("Search error")
        await message.answer("⚠️ Ошибка при поиске.", reply_markup=kb_home())


@router.callback_query(F.data == "do_search_fresh")
async def cb_search_fresh(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("🔍 Отправь текст или голосовое — я найду в памяти.")


# ═══════════════════════════════════════════════════════════════
#  ASK AI (notes context + general knowledge)
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "do_ask_ai")
async def cb_ask_ai(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    data = _pending.pop(user_id, None)
    if not data:
        await callback.answer("Текст не найден, отправь сообщение ещё раз.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await _perform_ask(user_id, data["text"], callback.message)


async def _perform_ask(user_id: int, text: str, message: Message) -> None:
    await message.answer("🤖 Думаю...")
    last_msg = _last_bot_message.get(user_id)
    try:
        answer = await ask_ai(user_id, text, last_bot_message=last_msg)
        _last_bot_message[user_id] = answer
        await message.answer(answer, reply_markup=kb_after_search())
    except Exception:
        logger.exception("Ask AI error")
        await message.answer("⚠️ Ошибка при обращении к ИИ.", reply_markup=kb_home())


@router.callback_query(F.data == "save_answer")
async def cb_save_answer(callback: CallbackQuery) -> None:
    """Сохранить последний ответ ИИ/поиска как заметку."""
    user_id = callback.from_user.id
    answer = _last_bot_message.get(user_id)
    if not answer:
        await callback.answer("Нет ответа для сохранения.")
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    try:
        note, reminders = await process_and_save_note(
            user_id=user_id, text=answer, note_type=NoteType.TEXT,
        )
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        rem_line = (
            f"\n⏰ Поставил напоминаний: {reminders} (напомню за сутки)"
            if reminders else ""
        )
        await callback.message.answer(
            f"✅ Ответ сохранён в заметки!\n\n"
            f"📋 {note.summary}\n"
            f"🏷 {tags_str}"
            f"{rem_line}",
            reply_markup=kb_after_save(),
        )
    except Exception:
        logger.exception("Error saving AI answer")
        await callback.message.answer("⚠️ Ошибка при сохранении.", reply_markup=kb_home())


@router.callback_query(F.data == "add_more")
async def cb_add_more(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("📝 Отправь текст или голосовое — я запомню.")


# ═══════════════════════════════════════════════════════════════
#  EDIT TAGS (last saved note — placeholder, uses edit flow)
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "edit_tags_last")
async def cb_edit_tags_last(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    await callback.answer()

    notes = await postgres.get_user_notes(user_id, limit=1, offset=0)
    if not notes:
        await callback.message.answer("Нет заметок для редактирования.", reply_markup=kb_home())
        return

    note = notes[0]
    tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
    await callback.message.answer(
        f"📝 Заметка:\n{note.summary}\n\n🏷 Теги: {tags_str}\n\n"
        "Для редактирования нажмите ✏️:",
        reply_markup=kb_note_actions(note.id),
    )


# ═══════════════════════════════════════════════════════════════
#  MANAGE MENU
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "manage_menu")
async def cb_manage_menu(callback: CallbackQuery) -> None:
    await callback.answer()
    count = await postgres.count_user_notes(callback.from_user.id)
    await callback.message.edit_text(
        f"📋 Управление заметками\n\nВсего заметок: {count}\n\nВыберите фильтр:",
        reply_markup=kb_manage_filters(),
    )


# ═══════════════════════════════════════════════════════════════
#  ALL NOTES (paginated)
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data.startswith("notes_all:"))
async def cb_notes_all(callback: CallbackQuery) -> None:
    await callback.answer()
    offset = int(callback.data.split(":")[1])
    user_id = callback.from_user.id

    notes = await postgres.get_user_notes(user_id, limit=NOTES_PER_PAGE, offset=offset)
    if not notes:
        await callback.message.edit_text("Заметок пока нет.", reply_markup=kb_home())
        return

    tz = await timezones.get_tz(user_id)
    await callback.message.edit_text(
        "📋 Ваши заметки:",
        reply_markup=_notes_list_kb(notes, offset, tz, "notes_all"),
    )


# ═══════════════════════════════════════════════════════════════
#  FILTERS
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "filter_date")
async def cb_filter_date(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(
        "📅 Фильтр по дате:\n\nВыберите период:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="Сегодня", callback_data="fdate:today"),
                InlineKeyboardButton(text="Эта неделя", callback_data="fdate:week"),
            ],
            [
                InlineKeyboardButton(text="Этот месяц", callback_data="fdate:month"),
                InlineKeyboardButton(text="Все", callback_data="notes_all:0"),
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="manage_menu")],
        ]),
    )


@router.callback_query(F.data.startswith("fdate:"))
async def cb_fdate(callback: CallbackQuery) -> None:
    await callback.answer()
    period = callback.data.split(":")[1]
    user_id = callback.from_user.id

    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)

    if period == "today":
        time_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        time_from = now - timedelta(days=now.weekday())
        time_from = time_from.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        time_from = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        time_from = None

    notes = await postgres.get_user_notes(user_id, time_from=time_from, limit=NOTES_PER_PAGE, offset=0)
    if not notes:
        await callback.message.edit_text("За этот период заметок нет.", reply_markup=kb_after_action())
        return

    tz = await timezones.get_tz(user_id)
    await callback.message.edit_text(
        f"📅 Заметки за период:",
        reply_markup=_notes_list_kb(notes, 0, tz, "notes_all"),
    )


@router.callback_query(F.data == "filter_tag")
async def cb_filter_tag(callback: CallbackQuery) -> None:
    await callback.answer()
    user_id = callback.from_user.id
    tags = await postgres.get_user_tags(user_id)

    if not tags:
        await callback.message.edit_text("У вас пока нет тегов.", reply_markup=kb_after_action())
        return

    buttons = []
    row = []
    for tag in tags[:20]:  # max 20 tags
        row.append(InlineKeyboardButton(text=f"#{tag}", callback_data=f"ftag:{tag}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="manage_menu")])

    await callback.message.edit_text("🏷 Выберите тег:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))


@router.callback_query(F.data.startswith("ftag:"))
async def cb_ftag(callback: CallbackQuery) -> None:
    await callback.answer()
    tag = callback.data.split(":", 1)[1]
    user_id = callback.from_user.id

    notes = await postgres.get_user_notes(user_id, tag=tag, limit=NOTES_PER_PAGE, offset=0)
    if not notes:
        await callback.message.edit_text(f"Заметок с тегом #{tag} нет.", reply_markup=kb_after_action())
        return

    tz = await timezones.get_tz(user_id)
    await callback.message.edit_text(
        f"🏷 Заметки с тегом #{tag}:",
        reply_markup=_notes_list_kb(notes, 0, tz, "notes_all"),
    )


@router.callback_query(F.data == "filter_type")
async def cb_filter_type(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(
        "📝 Тип заметки:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📝 Текст", callback_data="ftype:text"),
                InlineKeyboardButton(text="🎤 Голос", callback_data="ftype:voice"),
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="manage_menu")],
        ]),
    )


@router.callback_query(F.data.startswith("ftype:"))
async def cb_ftype(callback: CallbackQuery) -> None:
    await callback.answer()
    note_type = callback.data.split(":")[1]
    user_id = callback.from_user.id

    notes = await postgres.get_user_notes(user_id, note_type=note_type, limit=NOTES_PER_PAGE, offset=0)
    if not notes:
        type_label = "текстовых" if note_type == "text" else "голосовых"
        await callback.message.edit_text(f"Нет {type_label} заметок.", reply_markup=kb_after_action())
        return

    tz = await timezones.get_tz(user_id)
    await callback.message.edit_text(
        "📋 Найденные заметки:",
        reply_markup=_notes_list_kb(notes, 0, tz, "notes_all"),
    )


@router.callback_query(F.data == "filter_kw")
async def cb_filter_kw(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(
        "🔍 Отправь ключевые слова для поиска — я найду совпадения в заметках.",
    )
    # Next text message will go through handle_text → shown as choice


# ═══════════════════════════════════════════════════════════════
#  NOTE ACTIONS
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data.startswith("note_actions:"))
async def cb_note_actions(callback: CallbackQuery) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]
    note = await postgres.get_note_by_id(note_id)

    if not note:
        await callback.message.edit_text("Заметка не найдена.", reply_markup=kb_after_action())
        return

    tz = await timezones.get_tz(callback.from_user.id)
    date_str = fmt_date(note.created_at, tz)
    type_labels = {NoteType.VOICE: "🎤 голос", NoteType.IMAGE: "🖼 фото"}
    type_icon = type_labels.get(note.note_type, "📝 текст")
    tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"

    await callback.message.edit_text(
        f"📋 Заметка от {date_str} ({type_icon})\n\n"
        f"📝 {note.summary}\n"
        f"🏷 {tags_str}",
        reply_markup=kb_note_actions(note.id),
    )


# ── View full note ──────────────────────────────────────────────

@router.callback_query(F.data.startswith("note_view:"))
async def cb_note_view(callback: CallbackQuery) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]
    note = await postgres.get_note_by_id(note_id)

    if not note:
        await callback.message.edit_text("Заметка не найдена.", reply_markup=kb_after_action())
        return

    tz = await timezones.get_tz(callback.from_user.id)
    date_str = fmt_date(note.created_at, tz)
    tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
    text = note.full_text[:3500]  # Telegram message limit

    await callback.message.edit_text(
        f"📋 Заметка от {date_str}\n\n"
        f"{text}\n\n"
        f"🏷 {tags_str}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✏️ Редактировать", callback_data=f"note_edit:{note_id}"),
                InlineKeyboardButton(text="🗑 Удалить", callback_data=f"note_del:{note_id}"),
            ],
            [
                InlineKeyboardButton(text="⬅️ Назад", callback_data="manage_menu"),
                InlineKeyboardButton(text="🏠 Главное меню", callback_data="home"),
            ],
        ]),
    )


# ── Edit note (FSM) ────────────────────────────────────────────

@router.callback_query(F.data.startswith("note_edit:"))
async def cb_note_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]
    note = await postgres.get_note_by_id(note_id)

    if not note:
        await callback.message.edit_text("Заметка не найдена.", reply_markup=kb_after_action())
        return

    await state.set_state(EditNote.waiting_for_text)
    await state.update_data(edit_note_id=note_id)

    await callback.message.edit_text(
        f"✏️ Текущий текст заметки:\n\n"
        f"{note.full_text[:3500]}\n\n"
        "Отправь новый текст заметки:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"note_actions:{note_id}")],
        ]),
    )


async def _finish_edit(message: Message, state: FSMContext, new_text: str) -> None:
    data = await state.get_data()
    note_id = data.get("edit_note_id")
    await state.clear()

    if not note_id:
        await message.answer("Ошибка: заметка для редактирования не найдена.", reply_markup=kb_home())
        return

    note = await postgres.get_note_by_id(note_id)
    if not note:
        await message.answer("Заметка не найдена.", reply_markup=kb_home())
        return

    try:
        updated, reminders = await update_existing_note(note, new_text)
        tags_str = ", ".join(f"#{t}" for t in updated.tags) if updated.tags else "—"
        rem_line = (
            f"\n⏰ Напоминаний: {reminders} (напомню за сутки)"
            if reminders else ""
        )
        await message.answer(
            f"✅ Заметка обновлена!\n\n"
            f"📋 {updated.summary}\n"
            f"🏷 {tags_str}"
            f"{rem_line}",
            reply_markup=kb_after_action(),
        )
    except Exception:
        logger.exception("Error updating note")
        await message.answer("⚠️ Ошибка при обновлении.", reply_markup=kb_home())


# ── Delete note ─────────────────────────────────────────────────

@router.callback_query(F.data.startswith("note_del:"))
async def cb_note_del(callback: CallbackQuery) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]
    await callback.message.edit_text(
        "🗑 Вы уверены, что хотите удалить эту заметку?",
        reply_markup=kb_confirm_delete(note_id),
    )


@router.callback_query(F.data.startswith("note_del_yes:"))
async def cb_note_del_yes(callback: CallbackQuery) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]

    try:
        await delete_note_full(note_id)
        await callback.message.edit_text(
            "✅ Заметка удалена.",
            reply_markup=kb_after_action(),
        )
    except Exception:
        logger.exception("Error deleting note")
        await callback.message.edit_text("⚠️ Ошибка при удалении.", reply_markup=kb_home())


@router.callback_query(F.data.startswith("note_del_no:"))
async def cb_note_del_no(callback: CallbackQuery) -> None:
    await callback.answer()
    note_id = callback.data.split(":", 1)[1]
    # Return to note actions
    note = await postgres.get_note_by_id(note_id)
    if note:
        tz = await timezones.get_tz(callback.from_user.id)
        date_str = fmt_date(note.created_at, tz)
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        await callback.message.edit_text(
            f"📋 Заметка от {date_str}\n\n📝 {note.summary}\n🏷 {tags_str}",
            reply_markup=kb_note_actions(note.id),
        )
    else:
        await callback.message.edit_text("Заметка не найдена.", reply_markup=kb_after_action())


# ═══════════════════════════════════════════════════════════════
#  REMINDERS
# ═══════════════════════════════════════════════════════════════

async def _reminders_view(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    reminders = await postgres.get_upcoming_reminders(user_id)
    tz = await timezones.get_tz(user_id)
    if not reminders:
        return (
            "⏰ Ближайших напоминаний нет.\n\n"
            "Запиши мысль с датой и временем — например «в среду в 19:00 театр», "
            "и я напомню за сутки.",
            kb_main_menu(),
        )
    lines = []
    buttons = []
    for r in reminders:
        when = fmt_date(r.event_at, tz, "%d.%m.%Y %H:%M")
        lines.append(f"📌 {when} — {r.title}")
        buttons.append([InlineKeyboardButton(
            text=f"🗑 {r.title[:35]}", callback_data=f"rem_del:{r.id}"
        )])
    buttons.append([InlineKeyboardButton(text="🏠 Главное меню", callback_data="home")])
    text = (
        "⏰ Ближайшие напоминания:\n\n"
        + "\n".join(lines)
        + "\n\nНажми «🗑», чтобы убрать ненужное."
    )
    return text, InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(Command("reminders"))
async def cmd_reminders(message: Message, state: FSMContext) -> None:
    await state.clear()
    text, kb = await _reminders_view(message.from_user.id)
    await message.answer(text, reply_markup=kb)


@router.callback_query(F.data == "reminders")
async def cb_reminders(callback: CallbackQuery) -> None:
    await callback.answer()
    text, kb = await _reminders_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


@router.callback_query(F.data.startswith("rem_del:"))
async def cb_rem_del(callback: CallbackQuery) -> None:
    await callback.answer("Напоминание убрано")
    rid = callback.data.split(":", 1)[1]
    await postgres.delete_reminder(rid)
    text, kb = await _reminders_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


# ═══════════════════════════════════════════════════════════════
#  WEEKLY SUMMARY
# ═══════════════════════════════════════════════════════════════

async def _send_summary(user_id: int, send) -> None:
    wait = await send("🗓 Готовлю сводку за неделю...")
    try:
        text = await build_weekly_summary(user_id)
        await wait.edit_text(text, reply_markup=kb_main_menu())
    except Exception:
        logger.exception("Summary error")
        await wait.edit_text("⚠️ Не удалось сформировать сводку.", reply_markup=kb_home())


@router.message(Command("summary"))
async def cmd_summary(message: Message, state: FSMContext) -> None:
    await state.clear()
    await _send_summary(message.from_user.id, message.answer)


@router.callback_query(F.data == "do_summary")
async def cb_summary(callback: CallbackQuery) -> None:
    await callback.answer()
    await _send_summary(callback.from_user.id, callback.message.answer)


# ═══════════════════════════════════════════════════════════════
#  SETTINGS (timezone + weekly summary schedule)
# ═══════════════════════════════════════════════════════════════

async def _settings_view(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    s = await postgres.get_user_settings(user_id) or UserSettings(user_id=user_id)
    status = "включена ✅" if s.summary_enabled else "выключена ❌"
    day = WEEKDAY_NAMES[s.summary_weekday]
    text = (
        "⚙️ Настройки\n\n"
        f"🌍 Часовой пояс: {s.timezone}\n"
        f"🗓 Авто-сводка: {status}\n"
        f"📅 День: {day}\n"
        f"🕐 Время: {s.summary_hour:02d}:00"
    )
    toggle_label = "🔕 Выключить сводку" if s.summary_enabled else "🔔 Включить сводку"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🌍 Сменить часовой пояс", callback_data="set_tz")],
        [InlineKeyboardButton(text=toggle_label, callback_data="sum_toggle")],
        [
            InlineKeyboardButton(text="📅 День сводки", callback_data="sum_day"),
            InlineKeyboardButton(text="🕐 Время сводки", callback_data="sum_hour"),
        ],
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="home")],
    ])
    return text, kb


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    await state.clear()
    text, kb = await _settings_view(message.from_user.id)
    await message.answer(text, reply_markup=kb)


@router.callback_query(F.data == "settings")
async def cb_settings(callback: CallbackQuery) -> None:
    await callback.answer()
    text, kb = await _settings_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


@router.callback_query(F.data == "set_tz")
async def cb_set_tz_menu(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text("🌍 Выбери часовой пояс:", reply_markup=kb_timezone())


async def _current_settings(user_id: int) -> UserSettings:
    return await postgres.get_user_settings(user_id) or UserSettings(user_id=user_id)


@router.callback_query(F.data == "sum_toggle")
async def cb_sum_toggle(callback: CallbackQuery) -> None:
    await callback.answer()
    s = await _current_settings(callback.from_user.id)
    await postgres.set_summary_schedule(
        s.user_id, s.summary_weekday, s.summary_hour, not s.summary_enabled
    )
    text, kb = await _settings_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


@router.callback_query(F.data == "sum_day")
async def cb_sum_day(callback: CallbackQuery) -> None:
    await callback.answer()
    buttons = [
        [InlineKeyboardButton(text=name, callback_data=f"sum_setday:{i}")]
        for i, name in enumerate(WEEKDAY_NAMES)
    ]
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="settings")])
    await callback.message.edit_text(
        "📅 В какой день присылать сводку?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("sum_setday:"))
async def cb_sum_setday(callback: CallbackQuery) -> None:
    await callback.answer()
    weekday = int(callback.data.split(":")[1])
    s = await _current_settings(callback.from_user.id)
    await postgres.set_summary_schedule(s.user_id, weekday, s.summary_hour, s.summary_enabled)
    text, kb = await _settings_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


@router.callback_query(F.data == "sum_hour")
async def cb_sum_hour(callback: CallbackQuery) -> None:
    await callback.answer()
    buttons = []
    row = []
    for h in range(24):
        row.append(InlineKeyboardButton(text=f"{h:02d}", callback_data=f"sum_sethour:{h}"))
        if len(row) == 6:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="settings")])
    await callback.message.edit_text(
        "🕐 В котором часу присылать сводку (по твоему времени)?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("sum_sethour:"))
async def cb_sum_sethour(callback: CallbackQuery) -> None:
    await callback.answer()
    hour = int(callback.data.split(":")[1])
    s = await _current_settings(callback.from_user.id)
    await postgres.set_summary_schedule(s.user_id, s.summary_weekday, hour, s.summary_enabled)
    text, kb = await _settings_view(callback.from_user.id)
    await callback.message.edit_text(text, reply_markup=kb)


# ═══════════════════════════════════════════════════════════════
#  ADMIN: /admin — user management
# ═══════════════════════════════════════════════════════════════

def _is_admin(user_id: int) -> bool:
    return user_id == settings.admin_user_id


def _admin_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="➕ Добавить пользователя", callback_data="admin_add"),
            InlineKeyboardButton(text="➖ Удалить пользователя", callback_data="admin_remove"),
        ],
        [
            InlineKeyboardButton(text="📋 Список пользователей", callback_data="admin_list"),
        ],
        [
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="home"),
        ],
    ])


@router.message(Command("admin"))
async def cmd_admin(message: Message, state: FSMContext) -> None:
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Только администратор может использовать эту команду.")
        return

    await state.clear()
    count = len(rt.allowed_ids)
    await message.answer(
        f"⚙️ Панель администратора\n\n"
        f"Пользователей с доступом: {count}",
        reply_markup=_admin_kb(),
    )


@router.callback_query(F.data == "admin_list")
async def cb_admin_list(callback: CallbackQuery) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.")
        return

    await callback.answer()
    if not rt.allowed_ids:
        text = "Список пуст — доступ открыт всем."
    else:
        lines = []
        for uid in sorted(rt.allowed_ids):
            marker = " (admin)" if uid == settings.admin_user_id else ""
            lines.append(f"• {uid}{marker}")
        text = "👥 Пользователи с доступом:\n\n" + "\n".join(lines)

    await callback.message.edit_text(text, reply_markup=_admin_kb())


@router.callback_query(F.data == "admin_add")
async def cb_admin_add(callback: CallbackQuery, state: FSMContext) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.")
        return

    await callback.answer()
    await state.set_state(AdminStates.waiting_add_user)
    await callback.message.edit_text(
        "Отправь Telegram ID пользователя для добавления:\n\n"
        "(Узнать ID можно через @userinfobot)",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="admin_cancel")],
        ]),
    )


@router.callback_query(F.data == "admin_remove")
async def cb_admin_remove(callback: CallbackQuery, state: FSMContext) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.")
        return

    await callback.answer()

    removable = sorted(uid for uid in rt.allowed_ids if uid != settings.admin_user_id)
    if not removable:
        await callback.message.edit_text(
            "Нет пользователей для удаления (кроме админа).",
            reply_markup=_admin_kb(),
        )
        return

    buttons = []
    for uid in removable:
        buttons.append([InlineKeyboardButton(text=f"🗑 {uid}", callback_data=f"admin_rm:{uid}")])
    buttons.append([InlineKeyboardButton(text="❌ Отмена", callback_data="admin_cancel")])

    await callback.message.edit_text(
        "Выбери пользователя для удаления:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("admin_rm:"))
async def cb_admin_rm(callback: CallbackQuery) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа.")
        return

    await callback.answer()
    uid = int(callback.data.split(":")[1])
    rt.allowed_ids.discard(uid)

    await callback.message.edit_text(
        f"✅ Пользователь {uid} удалён из списка доступа.",
        reply_markup=_admin_kb(),
    )


@router.callback_query(F.data == "admin_cancel")
async def cb_admin_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer()
    count = len(rt.allowed_ids)
    await callback.message.edit_text(
        f"⚙️ Панель администратора\n\nПользователей с доступом: {count}",
        reply_markup=_admin_kb(),
    )


# Handle user ID input for adding
async def _finish_add_user(message: Message, state: FSMContext, text: str) -> None:
    if not _is_admin(message.from_user.id):
        return

    try:
        uid = int(text)
    except ValueError:
        await message.answer("⚠️ Отправь числовой Telegram ID.")
        return

    rt.allowed_ids.add(uid)
    await state.clear()
    await message.answer(
        f"✅ Пользователь {uid} добавлен.",
        reply_markup=_admin_kb(),
    )


# ═══════════════════════════════════════════════════════════════
#  ADMIN: /api — change OpenAI API key
# ═══════════════════════════════════════════════════════════════

@router.message(Command("api"))
async def cmd_api(message: Message, state: FSMContext) -> None:
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Только администратор может использовать эту команду.")
        return

    current_masked = rt.api_key[:8] + "..." + rt.api_key[-4:]
    await state.set_state(AdminStates.waiting_api_key)
    await message.answer(
        f"🔑 Текущий API-ключ: {current_masked}\n\n"
        "Отправь новый API-ключ OpenAI:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="admin_cancel")],
        ]),
    )


async def _finish_api_key(message: Message, state: FSMContext, text: str) -> None:
    if not _is_admin(message.from_user.id):
        return

    new_key = text.strip()
    if not new_key.startswith("sk-"):
        await message.answer("⚠️ API-ключ должен начинаться с 'sk-'. Попробуй ещё раз.")
        return

    rt.api_key = new_key
    reset_openai_client()
    await state.clear()

    masked = new_key[:8] + "..." + new_key[-4:]
    await message.answer(
        f"✅ API-ключ обновлён: {masked}\n\n"
        "⚠️ Ключ изменён только в памяти. После перезапуска бота вернётся ключ из .env.",
        reply_markup=_admin_kb(),
    )

