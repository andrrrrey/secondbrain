"""Standalone script to initialize database tables and Qdrant collection."""

import asyncio

from app.db import postgres, qdrant


async def main() -> None:
    print("Initializing PostgreSQL tables...")
    await postgres.init_tables()
    print("Initializing Qdrant collection...")
    qdrant.init_collection()
    print("Done!")
    await postgres.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
