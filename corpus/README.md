# Test corpus

A round-trip QA pipeline is only as good as the text you test it with. Clean news
prose passes almost everything; the interesting failures sit in numbers,
abbreviations, names, verse and old spelling. This corpus is built so that **every
stage of the pipeline gets exercised**, including the cases it is blind to.

| Part | Files | Chunks | What it stresses |
|---|---|---:|---|
| Curated edge cases (DE) | `edge_cases/de.yaml` | 33 | numbers, money, dates, abbreviations, acronyms, homophones, homographs, compounds, ß, foreign words, symbols |
| Curated edge cases (EN) | `edge_cases/en.yaml` | 20 | same for English, plus Roman numerals, British place names, Carroll's nonsense words |
| Retail domain (DE) | `domain/drogerie.de.yaml` | 12 | product texts as a voice assistant would read them: prices, base prices, dosage, INCI ingredient lists, safety notes |
| Literature (DE) | `literature/kafka_verwandlung.de.txt` | 33 | long nested sentences, guillemets, `--` dashes, old spelling (*ungeheueren*, *schloß*) |
| Literature (DE, verse) | `literature/busch_max_und_moritz.de.txt` | 21 | rhyme and metre, archaic words (*Pfühle*, *Müh'*), onomatopoeia (*Kikeriki! Kikikerikih!!*) |
| Literature (EN) | `literature/carroll_alice.en.txt` | 27 | the original use case (English audiobooks), plus the homophone puns *tale/tail* and *not/knot* |

## The `expect` field

Every curated item states which stage *should* settle it:

- **`stage1`**: normalization alone makes text and transcript identical (WER = 0).
  For example `2,95 €` vs. `zwei Euro fünfundneunzig`.
- **`stage2`**: a difference remains that the reviewer must explain. Homophones
  (*das/dass*), split compounds, phone numbers, acronyms spelled out or not.
- **`blind`**: the round trip **cannot** detect the problem, because the error
  produces the same transcript. Examples are wrong stress on homographs (*modern*
  the verb vs. the adjective, *umfahren*, *read/read*), meaning carried only by
  punctuation (*Komm, wir essen, Opa!*), or vowel length hidden by `ß → ss`
  normalization (*Maßen/Massen*). These items document the method's limits. They
  are cases for an audio-capable reviewer (`READBACK_ATTACH_AUDIO=true`) or a human,
  and they should not be read as a pass.

## Why these texts

- **Synthetic edge cases** give dense, targeted coverage. One sentence can test a
  currency format, an abbreviation and a homophone at once.
- **Real literature** keeps the evaluation honest. Natural sentence lengths,
  dialogue and punctuation produce the chunking and prosody problems that
  hand-written test sentences miss.
- **Retail texts** reflect a realistic production setting: an assistant reading
  product information aloud, where *"2 × täglich"* misread as *"12 täglich"* is
  a real problem rather than a cosmetic one.
- **German first, English second.** German compounds, inflected numerals
  (*dritte/dritten*) and the ß make normalization much harder than in English.

## Ground truth for evaluation

With the fake engines (`readback eval`), labelled defects are injected into the
"spoken" text: deletion, insertion, substitution, repetition, truncation and word
swap. Benign transcription noise is added too: homophone spelling, split compounds
and dropped punctuation. Because each change is recorded, the report shows
defect recall and the false-escalation rate per run.

## Sources and licences

The literature excerpts are in the public domain and come from Project Gutenberg
plain-text editions, with the Project Gutenberg header and licence removed as
its terms allow:

- Franz Kafka, *Die Verwandlung* (Kurt Wolff, 1917), [eBook #22367](https://www.gutenberg.org/ebooks/22367), opening of chapter I
- Wilhelm Busch, *Max und Moritz* (53rd ed., 1906), [eBook #17161](https://www.gutenberg.org/ebooks/17161), preface and first prank
- Lewis Carroll, *Alice's Adventures in Wonderland* (1865), [eBook #11](https://www.gutenberg.org/ebooks/11), opening of chapter I and the "tale/tail" exchange from chapter III

The edge-case and retail texts were written for this project. All brand and
product names in them are fictional.
