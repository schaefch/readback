"""Synthetic, labelled errors for offline runs and the evaluation harness.

Two families:

* **TTS faults** change *what is spoken* - they are real defects the pipeline must
  catch (deletions, hallucinated insertions, loops, truncation, ...).
* **STT noise** changes only *how a correct recording is written down* - benign
  differences the pipeline must *not* escalate to a human (homophones, split
  compounds, missing punctuation).

Because every injected change is recorded, a run with fake providers yields ground
truth for precision/recall of each review stage.
"""

from __future__ import annotations

import random
import re
from typing import Literal

from pydantic import BaseModel

from readback.lexicon import HOMOPHONES, Lang

TTSFaultKind = Literal["deletion", "insertion", "substitution", "repetition", "truncation", "swap"]
STTNoiseKind = Literal["homophone", "compound_split", "punctuation"]

TTS_FAULT_KINDS: tuple[TTSFaultKind, ...] = (
    "deletion",
    "insertion",
    "substitution",
    "repetition",
    "truncation",
    "swap",
)

# Plausible "wrong word" substitutions: similar-sounding but different meaning.
_CONFUSIONS = {
    "de": ["nicht", "nichts", "mich", "dich", "ihn", "ihm", "einen", "keinen", "Haus", "Maus"],
    "en": ["not", "now", "then", "than", "him", "her", "this", "that", "cat", "hat"],
}
_FILLERS = {"de": ["äh", "also", "hm"], "en": ["uh", "um", "well"]}


class Change(BaseModel):
    kind: str
    detail: str


def _word_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\S+", text))


def _strip(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word)


def inject_tts_fault(
    text: str, lang: Lang, rng: random.Random, kind: TTSFaultKind | None = None
) -> tuple[str, Change]:
    words = [m.group(0) for m in _word_spans(text)]
    if len(words) < 4:
        kind = "substitution" if kind not in (None, "substitution", "deletion") else kind
    kind = kind or rng.choice(TTS_FAULT_KINDS)
    n = len(words)

    if kind == "deletion":
        size = min(rng.randint(1, 3), max(1, n - 1))
        start = rng.randrange(0, n - size + 1)
        removed = words[start : start + size]
        words = words[:start] + words[start + size :]
        return " ".join(words), Change(kind=kind, detail=f"dropped '{' '.join(removed)}'")
    if kind == "insertion":
        pos = rng.randrange(0, n + 1)
        extra = rng.choice(_FILLERS[lang] + [_strip(w) for w in words if _strip(w)])
        words.insert(pos, extra)
        return " ".join(words), Change(kind=kind, detail=f"inserted '{extra}' at {pos}")
    if kind == "substitution":
        pos = rng.randrange(0, n)
        original = _strip(words[pos]) or words[pos]
        candidates = [c for c in _CONFUSIONS[lang] if c.lower() != original.lower()]
        replacement = rng.choice(candidates)
        words[pos] = words[pos].replace(original, replacement) if original else replacement
        return " ".join(words), Change(kind=kind, detail=f"'{original}' -> '{replacement}'")
    if kind == "repetition":
        size = min(rng.randint(2, 4), n)
        start = rng.randrange(0, n - size + 1)
        span = words[start : start + size]
        words = words[: start + size] + span + words[start + size :]
        return " ".join(words), Change(kind=kind, detail=f"repeated '{' '.join(span)}'")
    if kind == "truncation":
        keep = max(1, int(n * rng.uniform(0.5, 0.8)))
        return " ".join(words[:keep]), Change(kind=kind, detail=f"cut after word {keep}/{n}")
    # swap
    pos = rng.randrange(0, n - 1)
    words[pos], words[pos + 1] = words[pos + 1], words[pos]
    return " ".join(words), Change(kind="swap", detail=f"swapped words {pos} and {pos + 1}")


def apply_stt_noise(text: str, lang: Lang, rng: random.Random) -> tuple[str, list[Change]]:
    """Rewrite a *correct* transcript in ways that a good reviewer should accept."""
    words = [m.group(0) for m in _word_spans(text)]
    changes: list[Change] = []

    homophone_positions = [
        i
        for i, w in enumerate(words)
        if any(_strip(w).lower() in group for group in HOMOPHONES[lang])
    ]
    if homophone_positions and rng.random() < 0.7:
        i = rng.choice(homophone_positions)
        core = _strip(words[i])
        group = next(g for g in HOMOPHONES[lang] if core.lower() in g)
        alt = rng.choice(sorted(group - {core.lower()}))
        alt = alt.capitalize() if core[:1].isupper() else alt
        words[i] = words[i].replace(core, alt)
        changes.append(Change(kind="homophone", detail=f"'{core}' -> '{alt}'"))

    long_words = [i for i, w in enumerate(words) if len(_strip(w)) >= 12 and _strip(w).isalpha()]
    if long_words and rng.random() < 0.5:
        i = rng.choice(long_words)
        core = _strip(words[i])
        cut = rng.randint(5, len(core) - 5)
        split = f"{core[:cut]} {core[cut:]}"
        words[i] = words[i].replace(core, split)
        changes.append(Change(kind="compound_split", detail=f"'{core}' -> '{split}'"))

    out = " ".join(words)
    if rng.random() < 0.5:
        out = re.sub(r"[,:](?!\d)|[;!?»«„“”\"]", "", out)  # keep "2,95" and "14:30"
        changes.append(Change(kind="punctuation", detail="dropped punctuation"))
    return out, changes
