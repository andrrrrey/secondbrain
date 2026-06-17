# AI Second Brain — Telegram Bot

Telegram-бот для хранения заметок и семантического поиска по ним.

## Стек

- **Bot**: aiogram 3.x
- **LLM**: OpenAI / OpenRouter
- **STT**: Whisper API
- **Vector DB**: Qdrant
- **DB**: PostgreSQL + asyncpg
- **Контейнеризация**: Docker Compose

## Быстрый старт

```bash
# 1. Скопировать конфиг
cp .env.example .env
# Заполнить .env своими ключами

# 2. Запустить всё
docker compose up -d

# 3. Применить миграции
docker compose exec bot python -m scripts.init_db
```

## Структура проекта

```
ai-second-brain/
├── app/
│   ├── bot/            # Хэндлеры и мидлвари бота
│   ├── config/         # Настройки приложения
│   ├── db/             # Работа с PostgreSQL и Qdrant
│   ├── models/         # Pydantic-модели данных
│   └── services/       # Бизнес-логика (LLM, STT, embeddings, search)
├── migrations/         # SQL-миграции
├── scripts/            # Утилиты (init_db и т.д.)
├── tests/              # Тесты
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Использование

- Отправь текст → бот сохранит как заметку с автотегами
- Отправь голосовое → бот транскрибирует и сохранит
- Задай вопрос → бот найдёт релевантные заметки и ответит

## Ограничение доступа

В `.env` указывается `ALLOWED_USER_IDS` — список Telegram ID через запятую.
Бот отвечает только пользователям из белого списка.
