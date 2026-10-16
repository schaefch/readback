"""Checkpointer selection: in-memory for tests, SQLite locally, Postgres in the cluster."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


@asynccontextmanager
async def open_checkpointer(url: str) -> AsyncIterator[BaseCheckpointSaver]:
    if url == "memory":
        yield InMemorySaver()
    elif url.startswith("sqlite:///"):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = Path(url.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            yield saver
    elif url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            await saver.setup()
            yield saver
    else:
        raise ValueError(f"unsupported checkpointer url: {url!r}")
