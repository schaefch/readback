"""Text normalization: bring written text and STT transcripts into one spoken form.

Both the reference text and the transcript go through the same function, so the
goal is not linguistic perfection but *consistency*: "2,95 €", "2,95 Euro" and
"zwei Euro fünfundneunzig" must all end up as the same token sequence.
Whatever normalization cannot reconcile is handed to the reviewer stage.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache

from num2words import num2words

from readback.lexicon import ABBREVIATIONS, CURRENCY_WORDS, UNITS, Lang

MONTHS_DE = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"

_TYPOGRAPHY = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": " ",
        "”": " ",
        "„": " ",
        "»": " ",
        "«": " ",
        "›": " ",
        "‹": " ",
        "–": " ",
        "—": " ",
        "…": " ",
        " ": " ",
        " ": " ",
    }
)


def _num(value: str | int | float, lang: Lang, to: str = "cardinal") -> str:
    words: str = num2words(value, lang=lang, to=to)
    if lang == "de":
        # "einhundert"/"eintausend" are usually spoken as "hundert"/"tausend".
        words = re.sub(r"^ein(hundert|tausend)", r"\1", words)
    return words


# German ordinals are inflected ("dritte", "dritten", "dritter"); compare stems only.
_ORDINAL_STEMS_DE = {
    num2words(i, lang="de", to="ordinal").removesuffix("e").replace("ß", "ss")
    for i in range(1, 101)
}
_ORDINAL_DE = re.compile(
    rf"\b({'|'.join(sorted(_ORDINAL_STEMS_DE, key=len, reverse=True))})(e[nmrs]?)\b"
)
_EIN_FORMS = {"eins", "eine", "einen", "einem", "einer", "eines"}
_ARTICLES_DE = (
    "der|die|das|den|dem|des|am|im|vom|zum|zur|beim|ab|bis|seit|ein|eine|einen|jeden|jede"
)


def _alternation(keys: list[str]) -> str:
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


@cache
def _abbreviations(lang: Lang) -> dict[str, str]:
    """Lexicon entries plus sentence-initial capitalized variants ("Zzgl.", "Ca.")."""
    table = dict(ABBREVIATIONS[lang])
    for key, spoken in ABBREVIATIONS[lang].items():
        table.setdefault(key[0].upper() + key[1:], spoken[0].upper() + spoken[1:])
    return table


@cache
def _patterns(lang: Lang) -> dict[str, re.Pattern[str]]:
    dec = "," if lang == "de" else r"\."
    thousands = r"\." if lang == "de" else ","
    number = rf"\d{{1,3}}(?:{thousands}\d{{3}})+(?:{dec}\d+)?|\d+(?:{dec}\d+)?"
    currency = _alternation(list(CURRENCY_WORDS[lang]))
    units = _alternation([u for u in UNITS[lang] if u not in CURRENCY_WORDS[lang]])
    return {
        "abbr": re.compile(rf"(?<![\w.])({_alternation(list(_abbreviations(lang)))})(?!\w)"),
        "cur_post": re.compile(rf"({number})\s?({currency})(?!\w)"),
        "cur_pre": re.compile(rf"({currency})\s?({number})"),
        "time": re.compile(r"\b(\d{1,2}):(\d{2})\b(\s?Uhr)?"),
        "time_dot_de": re.compile(r"\b(\d{1,2})\.(\d{2})(?=\s?Uhr\b)"),
        "phone": re.compile(r"(?<![\d.,])0\d{2,}(?:[ /-]\d+)*"),
        "numdate_de": re.compile(
            r"\b(\d{1,2})\.(0?[1-9]|1[0-2])(?:\.(\d{4}|\d{2})?|(?=[\s,;)]|$))(?![\d])"
        ),
        "date_de": re.compile(rf"\b(\d{{1,2}})\.\s?({MONTHS_DE})\b"),
        "ordinal_de": re.compile(r"\b(\d{1,3})\.(?=\s+[a-zäöüß])"),
        "ordinal_art_de": re.compile(rf"\b({_ARTICLES_DE})\s(\d{{1,3}})\.(?=\s)", re.IGNORECASE),
        "unit": re.compile(rf"({number})\s?({units})(?![\w/])"),
        "ordinal_en": re.compile(r"\b(\d+)(?:st|nd|rd|th)\b"),
        "year": re.compile(r"(?<![\d.,])(1[1-9]\d\d)(?![\d.,]\d)"),
        "number": re.compile(number),
    }


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ROMAN_RE = re.compile(r"\b(?=[IVXLC]{2,}\b)C{0,3}(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b")


def _roman(numeral: str) -> int:
    total = 0
    for a, b in zip(numeral, numeral[1:] + " ", strict=True):
        value = _ROMAN[a]
        total += -value if b != " " and _ROMAN[b] > value else value
    return total


def _parse_number(raw: str, lang: Lang) -> int | float:
    if lang == "de":
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    return float(raw) if "." in raw else int(raw)


def _speak_number(raw: str, lang: Lang) -> str:
    value = _parse_number(raw, lang)
    if isinstance(value, float):
        # Read decimals digit by digit ("zwei Komma neun fünf"), as narrators do.
        integer, _, frac = f"{value}".partition(".")
        sep = "Komma" if lang == "de" else "point"
        digits = " ".join(_num(int(d), lang) for d in frac)
        return f"{_num(int(integer), lang)} {sep} {digits}"
    return _num(value, lang)


def _speak_currency(raw: str, symbol: str, lang: Lang) -> str:
    major, minor = CURRENCY_WORDS[lang][symbol]
    value = _parse_number(raw, lang)
    whole = int(value)
    cents = round((value - whole) * 100) if isinstance(value, float) else 0
    spoken = f"{_num(whole, lang)} {major}"
    if cents:
        spoken += f" {_num(cents, lang)}" if lang == "de" else f" {_num(cents, lang)} {minor}"
    return spoken


def expand(text: str, lang: Lang) -> str:
    """Expand abbreviations, currencies, units, times and numbers into words."""
    p = _patterns(lang)
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    text = re.sub(r"\[Illustration[^\]]*\]", " ", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)  # Gutenberg _italics_
    text = text.replace("--", " ")
    text = re.sub(r"(?<=\d)\s?[×x](?=\s)|\s×\s", " mal " if lang == "de" else " times ", text)

    text = p["abbr"].sub(lambda m: _abbreviations(lang)[m.group(1)], text)
    if lang == "en":  # "Chapter IV", "Henry VIII"; single "I" is the pronoun
        text = _ROMAN_RE.sub(lambda m: str(_roman(m.group(0))), text)
    text = p["cur_post"].sub(lambda m: _speak_currency(m.group(1), m.group(2), lang), text)
    text = p["cur_pre"].sub(lambda m: _speak_currency(m.group(2), m.group(1), lang), text)

    def _time(m: re.Match[str]) -> str:
        hours, minutes = int(m.group(1)), int(m.group(2))
        if lang == "de":
            spoken = f"{_num(hours, lang)} Uhr"
            return spoken + (f" {_num(minutes, lang)}" if minutes else "")
        return f"{_num(hours, lang)} " + (_num(minutes, lang) if minutes else "o'clock")

    # Numbers with a leading zero (phone numbers, article numbers) are read digit by
    # digit and their grouping is arbitrary, so compare them as one digit sequence.
    text = p["phone"].sub(
        lambda m: " ".join(_num(int(d), lang) for d in re.sub(r"\D", "", m.group(0))), text
    )
    if lang == "de":
        text = p["time_dot_de"].sub(lambda m: f"{m.group(1)}:{m.group(2)}", text)
    text = p["time"].sub(_time, text)
    if lang == "de":

        def _numdate(m: re.Match[str]) -> str:
            day, month = (
                _num(int(m.group(1)), lang, "ordinal"),
                _num(int(m.group(2)), lang, "ordinal"),
            )
            year = f" {_num(int(m.group(3)), lang, 'year')}" if m.group(3) else ""
            return f"{day} {month}{year}"

        text = p["numdate_de"].sub(_numdate, text)
        text = p["date_de"].sub(
            lambda m: f"{_num(int(m.group(1)), lang, 'ordinal')} {m.group(2)}", text
        )
        text = p["ordinal_art_de"].sub(
            lambda m: f"{m.group(1)} {_num(int(m.group(2)), lang, 'ordinal')}", text
        )
        text = p["ordinal_de"].sub(lambda m: _num(int(m.group(1)), lang, "ordinal"), text)
    else:
        text = p["ordinal_en"].sub(lambda m: _num(int(m.group(1)), lang, "ordinal"), text)
    text = p["unit"].sub(
        lambda m: f"{_speak_number(m.group(1), lang)} {UNITS[lang][m.group(2)]}", text
    )
    text = p["year"].sub(lambda m: _num(int(m.group(1)), lang, "year"), text)
    text = p["number"].sub(lambda m: _speak_number(m.group(0), lang), text)
    text = text.replace("&", " und " if lang == "de" else " and ")
    return text


def normalize(text: str, lang: Lang) -> list[str]:
    """Return the comparison tokens for ``text``."""
    text = expand(text, lang).lower()
    if lang == "de":
        text = text.replace("ß", "ss")  # old/new orthography ("muß" vs. "muss")
        text = _ORDINAL_DE.sub(r"\1", text)
    text = text.replace("'", "")
    text = re.sub(r"[^\w]+", " ", text)  # punctuation, hyphens, slashes
    tokens = text.split()
    if lang == "de":
        # "1 Tablette" is read "eine Tablette"; numeral and article sound alike.
        tokens = ["ein" if t in _EIN_FORMS else t for t in tokens]
    return tokens
