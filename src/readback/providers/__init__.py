"""Provider registry: build engines from settings, importing heavy deps lazily."""

from __future__ import annotations

from readback.config import Settings
from readback.providers.base import AudioArtifact, STTProvider, Transcript, TTSProvider

__all__ = ["AudioArtifact", "STTProvider", "TTSProvider", "Transcript", "build_stt", "build_tts"]

_EXTRA_HINT = "Install the optional dependencies: pip install -e '.[{extra}]'"


def build_tts(settings: Settings) -> TTSProvider:
    match settings.tts_provider:
        case "fake":
            from readback.providers.fake import FakeTTS

            return FakeTTS(
                fault_rate=settings.fake_fault_rate,
                sticky_fault_rate=settings.fake_sticky_fault_rate,
                seed=settings.seed,
            )
        case "piper":
            from readback.providers.local import PiperTTS

            try:
                return PiperTTS(settings.piper_voices, settings.piper_voices_dir)
            except ImportError as e:
                raise RuntimeError(_EXTRA_HINT.format(extra="local")) from e
        case "openai":
            from readback.providers.openai_compat import OpenAITTS

            return OpenAITTS(
                settings.openai_tts_model,
                settings.openai_tts_voice,
                settings.openai_tts_instructions,
            )
    raise ValueError(f"unknown TTS provider {settings.tts_provider!r}")


def build_stt(settings: Settings) -> STTProvider:
    match settings.stt_provider:
        case "fake":
            from readback.providers.fake import FakeSTT

            return FakeSTT(noise_rate=settings.fake_noise_rate, seed=settings.seed)
        case "faster_whisper":
            from readback.providers.local import FasterWhisperSTT

            try:
                return FasterWhisperSTT(
                    settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
                )
            except ImportError as e:
                raise RuntimeError(_EXTRA_HINT.format(extra="local")) from e
        case "openai":
            from readback.providers.openai_compat import OpenAISTT

            return OpenAISTT(settings.openai_stt_model)
    raise ValueError(f"unknown STT provider {settings.stt_provider!r}")
