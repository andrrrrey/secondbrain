from __future__ import annotations

import logging
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1536  # text-embedding-3-small

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info("Qdrant client created")
    return _client


def init_collection() -> None:
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection '%s' created", settings.qdrant_collection)

    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="user_id",
        field_schema=PayloadSchemaType.INTEGER,
    )
    logger.info("Qdrant payload indexes ensured")


def upsert_vector(note_id: str, user_id: int, created_at: datetime, vector: list[float]) -> None:
    client = get_client()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=note_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "created_at": created_at.timestamp(),
                },
            )
        ],
    )
    logger.info("Qdrant upsert: note_id=%s, user_id=%s", note_id, user_id)


def delete_vector(note_id: str) -> None:
    client = get_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=[note_id]),
    )
    logger.info("Qdrant delete: note_id=%s", note_id)


def search_similar(
    query_vector: list[float],
    user_id: int,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 5,
) -> list[tuple[str, float]]:
    """Returns list of (note_id, score)."""
    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]
    if time_from or time_to:
        range_kwargs = {}
        if time_from:
            range_kwargs["gte"] = time_from.timestamp()
        if time_to:
            range_kwargs["lte"] = time_to.timestamp()
        must_conditions.append(
            FieldCondition(key="created_at", range=Range(**range_kwargs))
        )

    client = get_client()

    count = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]),
        exact=True,
    )
    logger.info("Qdrant search: user_id=%s, total points for user=%s", user_id, count.count)

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=limit,
    )
    logger.info("Qdrant results: %s", [(hit.id, round(hit.score, 3)) for hit in results])
    return [(hit.id, hit.score) for hit in results]
