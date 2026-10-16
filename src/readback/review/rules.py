"""Deterministic first-line reviewer: cheap, explainable, no network.

It settles the clear-cut cases (homophones, split compounds, truncation, loops) and
marks everything else as ``unclear`` so the LLM or a human can take over.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from langchain_core.runnables import RunnableConfig

from readback.lexicon import Lang, are_homophones
from readback.review.models import OpAssessment, ReviewInput, ReviewRecord
from readback.scoring import DiffOp, Score

# Phrases Whisper is known to hallucinate on silence or noise.
WHISPER_HALLUCINATIONS = {
    "de": [["untertitel", "im", "auftrag", "des", "zdf"], ["vielen", "dank", "fürs", "zuschauen"]],
    "en": [["thanks", "for", "watching"], ["thank", "you", "for", "watching"]],
}
FILLERS = {"de": {"äh", "ähm", "hm", "also"}, "en": {"uh", "um", "hmm", "well"}}

# British -> American spelling (Gutenberg texts are British, Whisper writes American).
_UK_US = [
    (r"our\b", "or"),
    (r"re\b", "er"),
    (r"is(e|ed|ing)\b", r"iz\1"),
    (r"ll(ed|ing)\b", r"l\1"),
]


def _american(word: str) -> str:
    for pattern, repl in _UK_US:
        word = re.sub(pattern, repl, word)
    return word


def assess(op: DiffOp, score: Score, lang: Lang) -> tuple[OpAssessment, float]:
    """Classify one diff op; returns the assessment and a confidence."""
    diff = op.describe()

    def out(category: str, explanation: str, confidence: float) -> tuple[OpAssessment, float]:
        return OpAssessment(diff=diff, category=category, explanation=explanation), confidence  # type: ignore[arg-type]

    if op.ref and op.hyp and "".join(op.ref) == "".join(op.hyp):
        return out("stt_artifact", "Same letters, different word boundaries (compound).", 0.95)

    if op.kind == "substitute" and len(op.ref) == len(op.hyp):
        pairs = list(zip(op.ref, op.hyp, strict=True))
        if all(r == h or are_homophones(r, h, lang) for r, h in pairs):
            return out("homophone", "Homophones sound identical; spelling is an STT choice.", 0.9)

    if op.kind == "delete" and op.ref_pos + len(op.ref) == len(score.ref_tokens):
        return out("tts_error", "The ending of the text is missing (truncated audio).", 0.9)

    if op.kind == "insert":
        before = score.ref_tokens[max(0, op.ref_pos - len(op.hyp)) : op.ref_pos]
        if op.hyp == before:
            return out("tts_error", "A phrase is repeated (TTS loop/stutter).", 0.9)
        if any(op.hyp == phrase for phrase in WHISPER_HALLUCINATIONS[lang]):
            return out("stt_artifact", "Known Whisper hallucination phrase.", 0.85)
        if set(op.hyp) <= FILLERS[lang]:
            return out("tts_error", "Filler word that is not in the text.", 0.7)
        return out("unclear", "Extra words: TTS hallucination or STT artifact.", 0.4)

    if op.kind == "delete":
        return out("tts_error", "Words missing in the middle of the audio.", 0.7)

    # substitution
    if lang == "en" and len(op.ref) == len(op.hyp):
        pairs = list(zip(op.ref, op.hyp, strict=True))
        if all(_american(r) == _american(h) for r, h in pairs):
            return out("stt_artifact", "British vs. American spelling.", 0.9)

    ref, hyp = " ".join(op.ref), " ".join(op.hyp)
    similarity = SequenceMatcher(None, ref, hyp).ratio()
    if len(op.ref) == len(op.hyp) == 1 and similarity >= 0.75:
        return out(
            "unclear", f"Similar spelling ({similarity:.0%}): inflection or old orthography?", 0.5
        )
    return out("tts_error", "Different words are spoken.", 0.75)


class RuleReviewer:
    name = "rules"

    def judge(self, item: ReviewInput) -> ReviewRecord:
        results = [assess(op, item.score, item.lang) for op in item.score.ops]
        issues = [a for a, _ in results]
        categories = {a.category for a in issues}
        if "tts_error" in categories:
            decision = "tts_error"
            confidence = max(c for a, c in results if a.category == "tts_error")
        elif "unclear" in categories:
            decision = "uncertain"
            confidence = min(c for _, c in results)
        else:
            decision = "accept"
            confidence = min((c for _, c in results), default=1.0)
        reasoning = "; ".join(f"{a.diff}: {a.explanation}" for a in issues) or "No differences."
        return ReviewRecord(
            decision=decision,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning=reasoning,
            issues=issues,
            reviewer=self.name,
            attempt=item.attempt,
        )

    async def review(self, item: ReviewInput, config: RunnableConfig | None = None) -> ReviewRecord:
        return self.judge(item)
