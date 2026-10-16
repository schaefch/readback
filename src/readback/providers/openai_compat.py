"""Cloud engines via the OpenAI audio API - works with OpenAI and Azure OpenAI.

Install with ``pip install -e '.[openai]'``. If ``AZURE_OPENAI_ENDPOINT`` is set the
Azure client is used (key or Entra ID token from the environment), otherwise OpenAI.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from readback.lexicon import Lang
from readback.providers.base import AudioArtifact, Transcript, wav_duration


def _client() -> Any:
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AsyncAzureOpenAI(api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"))
    return AsyncOpenAI()


class OpenAITTS:
    name = "openai"

    def __init__(self, model: str, voice: str, instructions: str | None = None):
        self.model, self.voice, self.instructions = model, voice, instructions
        self._client = _client()

    async def synthesize(
        self, text: str, *, lang: Lang, out_path: Path, attempt: int
    ) -> AudioArtifact:
        path = out_path.with_suffix(".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {"instructions": self.instructions} if self.instructions else {}
        async with self._client.audio.speech.with_streaming_response.create(
            model=self.model, voice=self.voice, input=text, response_format="wav", **extra
        ) as response:
            await response.stream_to_file(path)
        return AudioArtifact(
            path=str(path),
            duration_s=wav_duration(path),
            provider=self.name,
            meta={"model": self.model, "voice": self.voice, "attempt": attempt},
        )


class OpenAISTT:
    name = "openai"

    def __init__(self, model: str):
        self.model = model
        self._client = _client()

    async def transcribe(self, audio: AudioArtifact, *, lang: Lang) -> Transcript:
        path = Path(audio.path)
        data = await asyncio.to_thread(path.read_bytes)
        result = await self._client.audio.transcriptions.create(
            model=self.model, file=(path.name, data), language=lang
        )
        return Transcript(text=result.text, provider=self.name, meta={"model": self.model})
