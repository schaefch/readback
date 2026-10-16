"""Provider interfaces. Pipeline nodes depend only on these protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from readback.lexicon import Lang


class AudioArtifact(BaseModel):
    path: str
    mime_type: str = "audio/wav"
    duration_s: float | None = None
    provider: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    text: str
    provider: str
    meta: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    async def synthesize(
        self, text: str, *, lang: Lang, out_path: Path, attempt: int
    ) -> AudioArtifact: ...


@runtime_checkable
class STTProvider(Protocol):
    name: str

    async def transcribe(self, audio: AudioArtifact, *, lang: Lang) -> Transcript: ...


def wav_duration(path: Path) -> float | None:
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except (wave.Error, OSError, EOFError):
        return None
