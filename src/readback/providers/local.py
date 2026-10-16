"""Local, offline engines: Piper (TTS) and faster-whisper (STT).

Install with ``pip install -e '.[local]'``. Both libraries are blocking, so calls run
in a worker thread; a lock serializes access to each loaded model.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import Any

from readback.lexicon import Lang
from readback.providers.base import AudioArtifact, Transcript, wav_duration


class PiperTTS:
    """One Piper voice per language, loaded on first use."""

    name = "piper"

    def __init__(self, voices: dict[str, str], voices_dir: Path):
        import piper  # noqa: F401 - fail fast if the extra is missing

        self.voices, self.voices_dir = voices, voices_dir
        self._loaded: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _voice(self, lang: Lang) -> Any:
        if lang not in self._loaded:
            from piper import PiperVoice

            name = self.voices[lang]
            model = Path(name) if name.endswith(".onnx") else self.voices_dir / f"{name}.onnx"
            if not model.exists():
                raise FileNotFoundError(
                    f"Piper voice {model} not found. Download it with: "
                    f"python -m piper.download_voices --download-dir {self.voices_dir} {name}"
                )
            self._loaded[lang] = PiperVoice.load(str(model))
        return self._loaded[lang]

    def _synthesize(self, text: str, lang: Lang, path: Path) -> None:
        with wave.open(str(path), "wb") as wf:
            self._voice(lang).synthesize_wav(text, wf)

    async def synthesize(
        self, text: str, *, lang: Lang, out_path: Path, attempt: int
    ) -> AudioArtifact:
        path = out_path.with_suffix(".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            await asyncio.to_thread(self._synthesize, text, lang, path)
        return AudioArtifact(
            path=str(path),
            duration_s=wav_duration(path),
            provider=self.name,
            meta={"voice": self.voices[lang], "attempt": attempt},
        )


class FasterWhisperSTT:
    name = "faster_whisper"

    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._lock = asyncio.Lock()

    def _transcribe(self, path: str, lang: Lang) -> str:
        segments, _info = self._model.transcribe(
            path,
            language=lang,
            beam_size=5,
            # Each chunk is independent; carrying context over invites hallucinations.
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments)

    async def transcribe(self, audio: AudioArtifact, *, lang: Lang) -> Transcript:
        async with self._lock:
            text = await asyncio.to_thread(self._transcribe, audio.path, lang)
        return Transcript(text=text, provider=self.name, meta={"model": self.model_size})
