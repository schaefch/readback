"""Language tools shared by the MCP server and the in-process LLM reviewer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from readback import lexicon, normalize, scoring
from readback.lexicon import Lang


def expand_for_speech(text: str, lang: Lang) -> str:
    """Rewrite numbers, currencies, units, times and abbreviations the way a
    narrator would say them, e.g. '2,95 €' -> 'zwei Euro fünfundneunzig'."""
    return normalize.expand(text, lang)


def normalize_text(text: str, lang: Lang) -> str:
    """Return the canonical comparison form (lowercase, spoken numbers, no
    punctuation) that the pipeline uses to compute the word error rate."""
    return " ".join(normalize.normalize(text, lang))


def lookup_abbreviation(token: str, lang: Lang) -> str:
    """Look up how a written abbreviation or unit (e.g. 'z. B.', 'ml', 'Dr.') is
    spoken. Returns an empty string if it is unknown."""
    return lexicon.lookup_abbreviation(token, lang) or ""


def check_homophones(word_a: str, word_b: str, lang: Lang) -> bool:
    """True if the two words sound identical in the given language (e.g.
    'das'/'dass', 'tale'/'tail'), so a swap is an STT spelling choice."""
    return lexicon.are_homophones(word_a, word_b, lang)


def compare_texts(reference: str, transcript: str, lang: Lang) -> dict[str, Any]:
    """Normalize both texts and return WER, CER and the list of word differences."""
    result = scoring.score(reference, transcript, lang)
    return {
        "wer": round(result.wer, 4),
        "cer": round(result.cer, 4),
        "differences": [op.describe() for op in result.ops],
    }


REVIEW_TOOLS: list[Callable[..., Any]] = [
    expand_for_speech,
    normalize_text,
    lookup_abbreviation,
    check_homophones,
    compare_texts,
]
REVIEW_TOOL_NAMES = {f.__name__ for f in REVIEW_TOOLS}
