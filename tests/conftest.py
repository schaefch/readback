from __future__ import annotations

from pathlib import Path

import pytest

from readback.config import Settings
from readback.providers.fake import FakeSTT, FakeTTS
from readback.review.rules import RuleReviewer
from readback.services import Services


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(audio_dir=tmp_path / "audio", checkpointer="memory", _env_file=None)  # type: ignore[call-arg]


def make_services(
    settings: Settings, *, fault=0.0, sticky=0.0, noise=0.0, reviewer=None
) -> Services:
    return Services(
        settings=settings,
        tts=FakeTTS(fault_rate=fault, sticky_fault_rate=sticky, seed=settings.seed),
        stt=FakeSTT(noise_rate=noise, seed=settings.seed),
        reviewer=reviewer or RuleReviewer(),
    )
