from __future__ import annotations

import logging
import tempfile
from pathlib import Path

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
from app.models import NoteType
from app.services.notes import delete_note_full, process_and_save_note, update_existing_note
from app.services.search import ask_ai, search_and_answer
from app.services.stt import transcribe

logger = logging.getLogger(__name__)
router = Router()

NOTES_PER_PAGE = 5

# ── FSM States ──────────────────────────────────────────────────

class EditNote(StatesGroup):
    waiting_for_text = State()


# ── Pending voice/text storage ──────────────────────────────────

_pending: dict[int, dict] = {}

# Last bot message per user for follow-up detection
_last_bot_message: dict[int, str] = {}

COMMANDS = [
    BotCommand(command="start", description="Главное меню"),
    BotCommand(command="help", description="Справка"),
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
    """After search result."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="➕ Добавить заметку", callback_data="add_more"),
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


def _notes_list_kb(notes, offset: int, prefix: str = "notes_all") -> InlineKeyboardMarkup:
    """Build paginated notes list with note buttons."""
    buttons = []
    for n in notes:
        date_str = n.created_at.strftime("%d.%m.%Y")
        type_icon = "🎤" if n.note_type == NoteType.VOICE else "📝"
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

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    _last_bot_message.pop(message.from_user.id, None)
    await message.answer(
        "👋 Привет! Расскажи свои мысли — я выслушаю и помогу структурировать.\n\n"
        "Отправь текст или голосовое сообщение.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu")],
        ]),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Как пользоваться:\n\n"
        "1️⃣ Отправь текст или голосовое\n"
        "2️⃣ Выбери: Запомнить или Найти в памяти\n"
        "3️⃣ Управляй заметками через меню\n"
    )


# ═══════════════════════════════════════════════════════════════
#  HOME callback
# ═══════════════════════════════════════════════════════════════

@router.callback_query(F.data == "home")
async def cb_home(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer()
    await callback.message.edit_text(
        "👋 Отправь текст или голосовое сообщение.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📋 Управление заметками", callback_data="manage_menu")],
        ]),
    )


# ═══════════════════════════════════════════════════════════════
#  INPUT: Voice → transcribe + choice
# ═══════════════════════════════════════════════════════════════

@router.message(lambda m: m.voice is not None)
async def handle_voice(message: Message, bot: Bot, state: FSMContext) -> None:
    # If we're editing a note, don't intercept
    current = await state.get_state()
    if current == EditNote.waiting_for_text:
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
#  INPUT: Text → choice (unless in FSM edit mode)
# ═══════════════════════════════════════════════════════════════

@router.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_text(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text:
        return

    current = await state.get_state()

    # If editing a note — handle FSM
    if current == EditNote.waiting_for_text:
        await _finish_edit(message, state, text)
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

    try:
        note = await process_and_save_note(
            user_id=user_id, text=data["text"], note_type=data["note_type"],
        )
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        await callback.message.answer(
            f"✅ Запомнил!\n\n"
            f"📋 {note.summary}\n"
            f"🏷 {tags_str}",
            reply_markup=kb_after_save(),
        )
    except Exception:
        logger.exception("Error saving note")
        await callback.message.answer("⚠️ Ошибка при сохранении.", reply_markup=kb_home())


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
    await callback.message.answer("🔍 Ищу в памяти...")

    try:
        answer = await search_and_answer(user_id, data["text"])
        _last_bot_message[user_id] = answer
        await callback.message.answer(answer, reply_markup=kb_after_search())
    except Exception:
        logger.exception("Search error")
        await callback.message.answer("⚠️ Ошибка при поиске.", reply_markup=kb_home())


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
    await callback.message.answer("🤖 Думаю...")

    query_text = data["text"]
    last_msg = _last_bot_message.get(user_id)

    try:
        answer = await ask_ai(user_id, query_text, last_bot_message=last_msg)
        _last_bot_message[user_id] = answer
        await callback.message.answer(answer, reply_markup=kb_after_search())
    except Exception:
        logger.exception("Ask AI error")
        await callback.message.answer("⚠️ Ошибка при обращении к ИИ.", reply_markup=kb_home())


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

    await callback.message.edit_text(
        "📋 Ваши заметки:",
        reply_markup=_notes_list_kb(notes, offset, "notes_all"),
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

    await callback.message.edit_text(
        f"📅 Заметки за период:",
        reply_markup=_notes_list_kb(notes, 0, "notes_all"),
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

    await callback.message.edit_text(
        f"🏷 Заметки с тегом #{tag}:",
        reply_markup=_notes_list_kb(notes, 0, "notes_all"),
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

    await callback.message.edit_text(
        "📋 Найденные заметки:",
        reply_markup=_notes_list_kb(notes, 0, "notes_all"),
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

    date_str = note.created_at.strftime("%d.%m.%Y %H:%M")
    type_icon = "🎤 голос" if note.note_type == NoteType.VOICE else "📝 текст"
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

    date_str = note.created_at.strftime("%d.%m.%Y %H:%M")
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
        updated = await update_existing_note(note, new_text)
        tags_str = ", ".join(f"#{t}" for t in updated.tags) if updated.tags else "—"
        await message.answer(
            f"✅ Заметка обновлена!\n\n"
            f"📋 {updated.summary}\n"
            f"🏷 {tags_str}",
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
        date_str = note.created_at.strftime("%d.%m.%Y %H:%M")
        tags_str = ", ".join(f"#{t}" for t in note.tags) if note.tags else "—"
        await callback.message.edit_text(
            f"📋 Заметка от {date_str}\n\n📝 {note.summary}\n🏷 {tags_str}",
            reply_markup=kb_note_actions(note.id),
        )
    else:
        await callback.message.edit_text("Заметка не найдена.", reply_markup=kb_after_action())
