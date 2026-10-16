import pytest

from readback.review.models import ReviewInput
from readback.review.rules import RuleReviewer
from readback.scoring import score


def judge(ref: str, hyp: str, lang: str = "de"):
    s = score(ref, hyp, lang)
    return RuleReviewer().judge(
        ReviewInput(chunk_id="c", lang=lang, reference=ref, transcript=hyp, score=s)
    )


def test_identical_after_normalization_has_zero_wer():
    s = score("Das kostet 2,95 €.", "das kostet zwei Euro fünfundneunzig", "de")
    assert s.wer == 0 and s.ops == []


def test_split_compound_is_one_op_and_accepted():
    ref = "Die Donaudampfschifffahrtsgesellschaft fährt."
    s = score(ref, "Die Donaudampf schifffahrtsgesellschaft fährt.", "de")
    assert len(s.ops) == 1 and s.ops[0].kind == "substitute"
    assert judge(ref, "Die Donaudampf schifffahrtsgesellschaft fährt.").decision == "accept"


@pytest.mark.parametrize(
    ("ref", "hyp", "lang"),
    [
        ("Ich weiß, dass es wahr ist.", "Ich weiß, das es war ist.", "de"),
        ("Mine is a long and a sad tale!", "Mine is a long and a sad tail!", "en"),
        (
            "It was labelled in the centre of the colour chart.",
            "It was labeled in the center of the color chart.",
            "en",
        ),
    ],
)
def test_spelling_only_differences_are_accepted(ref, hyp, lang):
    assert judge(ref, hyp, lang).decision == "accept"


@pytest.mark.parametrize(
    ("hyp", "expected"),
    [
        ("Er lag auf seinem panzerartig harten", "tts_error"),  # truncation
        ("Er lag auf seinem seinem panzerartig harten Rücken.", "tts_error"),  # repetition
        ("Er lag nicht auf seinem panzerartig harten Rücken.", "uncertain"),  # STT or TTS?
        ("Er lag auf panzerartig harten Rücken.", "tts_error"),  # mid deletion
        ("Er saß auf seinem panzerartig harten Rücken.", "tts_error"),  # wrong word
    ],
)
def test_rule_reviewer_flags_tts_defects(hyp, expected):
    assert judge("Er lag auf seinem panzerartig harten Rücken.", hyp).decision == expected


def test_written_abbreviation_in_transcript_goes_to_the_next_stage():
    assert judge("Das ist, d. h. heute, gut.", "Das ist dh heute gut.").decision == "uncertain"


def test_similar_spelling_is_left_to_the_next_stage():
    verdict = judge("zu einem ungeheueren Ungeziefer", "zu einem ungeheuren Ungeziefer")
    assert verdict.decision == "uncertain"
