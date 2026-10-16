import random

from readback.corpus import chunks_from_text, load_corpus, split_sentences
from readback.faults import TTS_FAULT_KINDS, apply_stt_noise, inject_tts_fault
from readback.normalize import normalize


def test_bundled_corpus_loads_and_respects_chunk_size():
    chunks = load_corpus(["corpus"], max_chars=400)
    assert len(chunks) > 100
    assert {c["lang"] for c in chunks} == {"de", "en"}
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert max(len(c["text"]) for c in chunks) <= 400


def test_sentence_split_keeps_abbreviations():
    parts = split_sentences("Bitte bringen Sie z. B. den Ausweis mit. Danke!", "de")
    assert parts == ["Bitte bringen Sie z. B. den Ausweis mit.", "Danke!"]


def test_chunks_from_text_splits_paragraphs():
    chunks = chunks_from_text("Erster Satz. Zweiter Satz.\n\nNeuer Absatz.", "de", 400)
    assert [c["text"] for c in chunks] == ["Erster Satz. Zweiter Satz.", "Neuer Absatz."]


def test_every_tts_fault_changes_the_spoken_words():
    text = "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich verwandelt."
    for kind in TTS_FAULT_KINDS:
        spoken, change = inject_tts_fault(text, "de", random.Random(1), kind)
        assert change.kind == kind
        assert normalize(spoken, "de") != normalize(text, "de"), kind


def test_stt_noise_keeps_numbers_intact():
    text = "Das kostet 2,95 € um 14:30 Uhr, und ich weiß, dass das wahr ist."
    for seed in range(30):
        noisy, _ = apply_stt_noise(text, "de", random.Random(seed))
        assert "2,95" in noisy and "14:30" in noisy
