import pytest

from readback.normalize import normalize


@pytest.mark.parametrize(
    ("written", "spoken", "lang"),
    [
        ("Das kostet 2,95 €.", "das kostet zwei Euro fünfundneunzig", "de"),
        ("2,95 Euro", "zwei Euro fünfundneunzig", "de"),
        ("z. B. 500 ml", "zum Beispiel fünfhundert Milliliter", "de"),
        ("Zzgl. Pfand", "zuzüglich Pfand", "de"),
        ("am 3. Mai um 14:30 Uhr", "am dritten Mai um vierzehn Uhr dreißig", "de"),
        ("der 2. Streich", "der zweite Streich", "de"),
        ("vom 12.05.2026", "vom zwölften fünften zweitausendsechsundzwanzig", "de"),
        ("seit 1987", "seit neunzehnhundertsiebenundachtzig", "de"),
        ("1.000 Gäste", "tausend Gäste", "de"),
        ("2 × täglich 1 Tablette", "zwei mal täglich eine Tablette", "de"),
        ("Ach, was muß man oft", "ach was muss man oft", "de"),
        (
            "»Was ist mit mir geschehen?« dachte er -- _Reisender_",
            "was ist mit mir geschehen dachte er reisender",
            "de",
        ),
        ("Mr. Smith paid $3.50", "mister smith paid three dollars fifty cents", "en"),
        ("on the 21st of May", "on the twenty-first of may", "en"),
        ("In 1865, 15% of them", "in eighteen sixty-five fifteen percent of them", "en"),
        ("the Rabbit-Hole", "the rabbit hole", "en"),
        ("Chapter IV and Henry VIII", "chapter four and henry eight", "en"),
        ("I think I can", "i think i can", "en"),
    ],
)
def test_written_and_spoken_forms_normalize_identically(written, spoken, lang):
    assert normalize(written, lang) == normalize(spoken, lang)


def test_real_differences_survive_normalization():
    assert normalize("Er kam nicht.", "de") != normalize("Er kam.", "de")
    assert normalize("2,95 €", "de") != normalize("2,59 €", "de")
