"""Dependencies the graph needs at runtime, passed as LangGraph ``context``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from readback.config import Settings, get_settings
from readback.providers import STTProvider, TTSProvider, build_stt, build_tts
from readback.review import Reviewer, build_reviewer


@dataclass
class Services:
    settings: Settings
    tts: TTSProvider
    stt: STTProvider
    reviewer: Reviewer

    @classmethod
    async def from_settings(cls, settings: Settings) -> Services:
        # Model loading (Piper, Whisper) blocks; keep the event loop responsive.
        tts, stt = await asyncio.gather(
            asyncio.to_thread(build_tts, settings), asyncio.to_thread(build_stt, settings)
        )
        return cls(settings=settings, tts=tts, stt=stt, reviewer=await build_reviewer(settings))


_default: Services | None = None
_lock = asyncio.Lock()


async def resolve(context: object) -> Services:
    """Use the services passed as context, or build defaults from the environment
    (e.g. when the graph runs inside LangGraph Studio / ``langgraph dev``)."""
    global _default
    if isinstance(context, Services):
        return context
    async with _lock:
        if _default is None:
            _default = await Services.from_settings(get_settings())
    return _default
