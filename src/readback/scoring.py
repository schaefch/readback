"""Compare normalized reference and transcript: WER, CER and the concrete edit ops."""

from __future__ import annotations

from typing import Literal

import jiwer
from pydantic import BaseModel

from readback.lexicon import Lang
from readback.normalize import normalize


class DiffOp(BaseModel):
    kind: Literal["substitute", "delete", "insert"]
    ref: list[str]
    hyp: list[str]
    ref_pos: int
    """Token index in the normalized reference where the op starts."""

    def describe(self) -> str:
        ref, hyp = " ".join(self.ref), " ".join(self.hyp)
        return {
            "substitute": f"'{ref}' -> '{hyp}'",
            "delete": f"missing '{ref}'",
            "insert": f"extra '{hyp}'",
        }[self.kind]


class Score(BaseModel):
    wer: float
    cer: float
    ref_tokens: list[str]
    hyp_tokens: list[str]
    ops: list[DiffOp]


def score(reference: str, transcript: str, lang: Lang) -> Score:
    ref, hyp = normalize(reference, lang), normalize(transcript, lang)
    if not ref:
        wer = 0.0 if not hyp else 1.0
        ops = [DiffOp(kind="insert", ref=[], hyp=hyp, ref_pos=0)] if hyp else []
        return Score(wer=wer, cer=wer, ref_tokens=ref, hyp_tokens=hyp, ops=ops)
    if not hyp:
        ops = [DiffOp(kind="delete", ref=ref, hyp=[], ref_pos=0)]
        return Score(wer=1.0, cer=1.0, ref_tokens=ref, hyp_tokens=hyp, ops=ops)

    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    # jiwer may report one edit as adjacent chunks (e.g. a split compound becomes
    # substitute + insert); merge touching chunks so each op is one coherent edit.
    spans: list[list[int]] = []
    for c in out.alignments[0]:
        if c.type == "equal":
            continue
        if spans and spans[-1][1] == c.ref_start_idx and spans[-1][3] == c.hyp_start_idx:
            spans[-1][1], spans[-1][3] = c.ref_end_idx, c.hyp_end_idx
        else:
            spans.append([c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx])
    ops = [
        DiffOp(
            kind="substitute" if r1 > r0 and h1 > h0 else ("delete" if r1 > r0 else "insert"),
            ref=ref[r0:r1],
            hyp=hyp[h0:h1],
            ref_pos=r0,
        )
        for r0, r1, h0, h1 in spans
    ]
    cer = jiwer.cer(" ".join(ref), " ".join(hyp))
    return Score(wer=out.wer, cer=float(cer), ref_tokens=ref, hyp_tokens=hyp, ops=ops)
