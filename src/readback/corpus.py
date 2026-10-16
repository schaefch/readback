"""Load test corpora and split them into chunks the TTS engine can handle.

Supported sources:

* ``*.yaml`` - curated items (``lang`` at top level, ``items: [{id, text, category}]``);
  each item is one chunk.
* ``*.<lang>.txt`` - running text; paragraphs are unwrapped and packed into chunks of
  whole sentences up to ``max_chars``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

from readback.lexicon import ABBREVIATIONS, Lang


class Chunk(TypedDict):
    id: str
    text: str
    lang: Lang
    source: str
    category: str
    expect: NotRequired[str]
    """Curated corpora: which stage should settle this chunk (stage1/stage2/blind)."""


def _protect_abbreviations(text: str, lang: Lang) -> str:
    for abbr in ABBREVIATIONS[lang]:
        text = text.replace(abbr, abbr.replace(".", "․"))  # one-dot leader
    return text


def split_sentences(text: str, lang: Lang) -> list[str]:
    protected = _protect_abbreviations(text, lang)
    parts = re.split(r"(?<=[.!?…])[»«“”\"’)]*\s+(?=[»«„“\"(]?[A-ZÄÖÜ0-9])", protected)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break an overlong sentence at clause boundaries (; : , and dashes)."""
    if len(sentence) <= max_chars:
        return [sentence]
    for pattern in (r"(?<=[;:])\s+", r"(?<=[,—])\s+|\s+(?=--)"):
        parts = re.split(pattern, sentence)
        if len(parts) > 1:
            return pack(parts, max_chars)
    return [sentence]


def pack(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in (piece for s in sentences for piece in _split_long(s, max_chars)):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _paragraphs(raw: str) -> list[str]:
    raw = re.sub(r"\[Illustration[^\]]*\]", "", raw)
    blocks = re.split(r"\n\s*\n", raw)
    paragraphs = [" ".join(line.strip() for line in b.splitlines() if line.strip()) for b in blocks]
    return [p for p in paragraphs if re.search(r"\w", p.replace("_", ""))]


def load_text(path: Path, max_chars: int) -> list[Chunk]:
    lang: Lang = "en" if path.name.endswith(".en.txt") else "de"
    chunks: list[Chunk] = []
    for paragraph in _paragraphs(path.read_text(encoding="utf-8")):
        if not paragraph:
            continue
        for text in pack(split_sentences(paragraph, lang), max_chars):
            chunks.append(
                Chunk(
                    id=f"{path.stem}-{len(chunks) + 1:04d}",
                    text=text,
                    lang=lang,
                    source=path.name,
                    category="literature",
                )
            )
    return chunks


def load_yaml(path: Path) -> list[Chunk]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    lang: Lang = data["lang"]
    return [
        Chunk(
            id=f"{path.stem}-{item['id']}",
            text=item["text"].strip(),
            lang=item.get("lang", lang),
            source=path.name,
            category=item.get("category", data.get("category", "misc")),
            expect=item.get("expect", ""),
        )
        for item in data["items"]
    ]


def load_corpus(paths: Sequence[str | Path], max_chars: int = 400) -> list[Chunk]:
    chunks: list[Chunk] = []
    for raw in paths:
        path = Path(raw)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.suffix in {".yaml", ".yml"}:
                chunks.extend(load_yaml(f))
            elif f.suffix == ".txt":
                chunks.extend(load_text(f, max_chars))
    return chunks


def chunks_from_text(text: str, lang: Lang, max_chars: int, source: str = "input") -> list[Chunk]:
    out: list[Chunk] = []
    for paragraph in _paragraphs(text):
        for piece in pack(split_sentences(paragraph, lang), max_chars):
            out.append(
                Chunk(
                    id=f"{source}-{len(out) + 1:04d}",
                    text=piece,
                    lang=lang,
                    source=source,
                    category="input",
                )
            )
    return out
