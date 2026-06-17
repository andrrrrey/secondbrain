import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from app.bot.handlers import router, set_bot_commands
from app.bot.middleware import AccessMiddleware
from app.config import settings
from app.db import postgres, qdrant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def on_startup(bot: Bot) -> None:
    await postgres.init_tables()
    qdrant.init_collection()
    await set_bot_commands(bot)
    logger.info("Startup complete")


async def on_shutdown() -> None:
    await postgres.close_pool()
    logger.info("Shutdown complete")


async def main() -> None:
    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.message.middleware(AccessMiddleware())
    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
