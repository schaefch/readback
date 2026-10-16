"""Deterministic offline providers for tests, CI and the evaluation harness.

``FakeTTS`` does not produce sound: it writes a small JSON "recording" holding the
text that *would* have been spoken (possibly with an injected fault).
``FakeSTT`` reads that file back (possibly with benign transcription noise).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path

from readback.faults import apply_stt_noise, inject_tts_fault
from readback.lexicon import Lang
from readback.normalize import normalize
from readback.providers.base import AudioArtifact, Transcript

WORDS_PER_SECOND = 2.6  # typical audiobook narration pace


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


class FakeTTS:
    """Simulated TTS.

    * ``fault_rate``: probability that one attempt contains a *transient* defect
      (re-synthesizing usually fixes it - typical for neural TTS).
    * ``sticky_fault_rate``: probability that a text triggers a *systematic* defect
      that survives every regeneration (e.g. a word the voice always mangles).
    """

    name = "fake"

    def __init__(self, fault_rate: float = 0.0, sticky_fault_rate: float = 0.0, seed: int = 0):
        self.fault_rate = fault_rate
        self.sticky_fault_rate = sticky_fault_rate
        self.seed = seed

    async def synthesize(
        self, text: str, *, lang: Lang, out_path: Path, attempt: int
    ) -> AudioArtifact:
        spoken, faults = text, []
        key = " ".join(normalize(text, lang))  # same for raw and "speakable" input
        sticky = _rng(self.seed, "sticky", key)
        if sticky.random() < self.sticky_fault_rate:
            spoken, change = inject_tts_fault(spoken, lang, sticky)
            faults.append(change.model_dump() | {"sticky": True})
        transient = _rng(self.seed, "transient", key, attempt)
        if transient.random() < self.fault_rate:
            spoken, change = inject_tts_fault(spoken, lang, transient)
            faults.append(change.model_dump() | {"sticky": False})

        path = out_path.with_suffix(".fake.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"spoken": spoken, "faults": faults}, ensure_ascii=False)
        await asyncio.to_thread(path.write_text, payload)
        return AudioArtifact(
            path=str(path),
            mime_type="application/json",
            duration_s=round(len(spoken.split()) / WORDS_PER_SECOND, 2),
            provider=self.name,
            meta={"spoken_text": spoken, "injected_faults": faults},
        )


class FakeSTT:
    name = "fake"

    def __init__(self, noise_rate: float = 0.0, seed: int = 0):
        self.noise_rate = noise_rate
        self.seed = seed

    async def transcribe(self, audio: AudioArtifact, *, lang: Lang) -> Transcript:
        spoken = json.loads(await asyncio.to_thread(Path(audio.path).read_text))["spoken"]
        rng = _rng(self.seed, "stt", spoken)
        noise = []
        if rng.random() < self.noise_rate:
            spoken, changes = apply_stt_noise(spoken, lang, rng)
            noise = [c.model_dump() for c in changes]
        return Transcript(text=spoken, provider=self.name, meta={"injected_noise": noise})
