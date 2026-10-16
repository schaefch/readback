"""Runtime configuration (12-factor: everything comes from env vars / ``.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="READBACK_", env_file=".env", extra="ignore")

    # --- engines -------------------------------------------------------------
    tts_provider: Literal["fake", "piper", "openai"] = "fake"
    stt_provider: Literal["fake", "faster_whisper", "openai"] = "fake"

    piper_voices: dict[str, str] = Field(
        default_factory=lambda: {"de": "de_DE-thorsten-medium", "en": "en_US-lessac-medium"}
    )
    piper_voices_dir: Path = Path("var/voices")
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_stt_model: str = "gpt-4o-transcribe"

    # Fake engines (offline demo / CI / evals)
    fake_fault_rate: float = 0.0
    fake_sticky_fault_rate: float = 0.0
    fake_noise_rate: float = 0.0
    seed: int = 0

    # --- review stage ----------------------------------------------------------
    reviewer: Literal["rules", "llm", "cascade"] = "rules"
    """``cascade`` = rules first, the LLM only for chunks the rules cannot settle."""
    llm_model: str = "anthropic:claude-opus-5-5"
    """Any ``init_chat_model`` id, e.g. ``azure_openai:<deployment>``."""
    llm_effort: Literal["low", "medium", "high"] | None = "low"
    """Anthropic effort level; review is a narrow task, so default to ``low``."""
    llm_use_mcp: bool = True
    """Load reviewer tools from the readback MCP server instead of in-process."""
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Extra MCP connections (langchain-mcp-adapters format) offered to the reviewer."""
    attach_audio: bool = False
    """Send the audio to the reviewer model (requires an audio-capable model)."""
    llm_prices_per_mtok: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "claude-opus-5-5": (4.0, 20.0),
            "claude-sonnet-5-5": (2.0, 10.0),
            "claude-haiku-5-5": (0.10, 0.50),
        }
    )

    # --- pipeline ----------------------------------------------------------------
    wer_threshold: float = 0.0
    max_regenerations: int = 2
    regeneration_strategy: Literal["retry", "speakable"] = "speakable"
    """``retry`` re-sends the same text (helps stochastic neural TTS); ``speakable``
    sends the text with numbers/abbreviations written out (helps deterministic TTS
    such as Piper, which would otherwise reproduce the same defect)."""
    max_concurrency: int = 4
    chunk_max_chars: int = 400
    audio_dir: Path = Path("var/audio")
    corpus_root: Path = Path("corpus")
    """API requests may only reference corpus files below this directory."""

    # --- service -----------------------------------------------------------------
    checkpointer: str = "sqlite:///var/checkpoints.sqlite"
    """``memory``, ``sqlite:///<path>`` or ``postgresql://...``."""
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
