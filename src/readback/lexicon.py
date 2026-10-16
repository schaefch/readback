"""Language resources used by normalization, the rule reviewer and the MCP server.

Everything here is deliberately plain data so it can be exposed 1:1 as MCP tools
and reviewed (or extended) without touching pipeline code.
"""

from __future__ import annotations

from typing import Literal

Lang = Literal["de", "en"]

# Abbreviations as they appear in written text -> how a narrator says them.
# Keys are matched case-sensitively and including the trailing dot.
ABBREVIATIONS: dict[str, dict[str, str]] = {
    "de": {
        "z. B.": "zum Beispiel",
        "z.B.": "zum Beispiel",
        "d. h.": "das heißt",
        "d.h.": "das heißt",
        "u. a.": "unter anderem",
        "u.a.": "unter anderem",
        "usw.": "und so weiter",
        "bzw.": "beziehungsweise",
        "ca.": "circa",
        "ggf.": "gegebenenfalls",
        "inkl.": "inklusive",
        "evtl.": "eventuell",
        "vgl.": "vergleiche",
        "Dr.": "Doktor",
        "Prof.": "Professor",
        "Nr.": "Nummer",
        "Str.": "Straße",
        "St.": "Sankt",
        "Mio.": "Millionen",
        "Mrd.": "Milliarden",
        "Tel.": "Telefon",
        "Hr.": "Herr",
        "Fr.": "Frau",
        "Jh.": "Jahrhundert",
        "zzgl.": "zuzüglich",
        "Art.-Nr.": "Artikelnummer",
    },
    "en": {
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miz",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "St.": "Saint",
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "et cetera",
        "vs.": "versus",
        "No.": "number",
        "approx.": "approximately",
        "a.m.": "a m",
        "p.m.": "p m",
    },
}

# Units / symbols that follow a number ("500 ml", "15 %").
UNITS: dict[str, dict[str, str]] = {
    "de": {
        "ml": "Milliliter",
        "l": "Liter",
        "mg": "Milligramm",
        "g": "Gramm",
        "kg": "Kilogramm",
        "mm": "Millimeter",
        "cm": "Zentimeter",
        "m": "Meter",
        "km": "Kilometer",
        "km/h": "Kilometer pro Stunde",
        "%": "Prozent",
        "°C": "Grad Celsius",
        "°": "Grad",
        "€": "Euro",
        "EUR": "Euro",
        "Stk.": "Stück",
        "min": "Minuten",
        "h": "Stunden",
    },
    "en": {
        "ml": "milliliters",
        "l": "liters",
        "mg": "milligrams",
        "g": "grams",
        "kg": "kilograms",
        "cm": "centimeters",
        "m": "meters",
        "km": "kilometers",
        "mph": "miles per hour",
        "%": "percent",
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
        "€": "euros",
        "$": "dollars",
        "£": "pounds",
        "min": "minutes",
        "h": "hours",
    },
}

CURRENCY_WORDS: dict[str, dict[str, tuple[str, str]]] = {
    # symbol -> (major unit, minor unit)
    # Written-out names are included because STT often emits "2,95 Euro".
    "de": {
        "€": ("Euro", "Cent"),
        "EUR": ("Euro", "Cent"),
        "Euro": ("Euro", "Cent"),
        "$": ("Dollar", "Cent"),
        "Dollar": ("Dollar", "Cent"),
    },
    "en": {
        "€": ("euros", "cents"),
        "euros": ("euros", "cents"),
        "$": ("dollars", "cents"),
        "dollars": ("dollars", "cents"),
        "£": ("pounds", "pence"),
        "pounds": ("pounds", "pence"),
    },
}

# Words that sound (near-)identical. A substitution inside one of these groups is
# almost always an STT spelling choice, not a TTS error.
HOMOPHONES: dict[str, list[frozenset[str]]] = {
    "de": [
        frozenset(s)
        for s in (
            {"das", "dass"},
            {"seid", "seit"},
            {"wider", "wieder"},
            {"wahr", "war"},
            {"viel", "fiel"},
            {"mehr", "meer"},
            {"malen", "mahlen"},
            {"lehre", "leere"},
            {"leib", "laib"},
            {"lied", "lid"},
            {"tod", "tot"},
            {"stadt", "statt"},
            {"wal", "wahl"},
            {"rad", "rat"},
            {"lerche", "lärche"},
            {"weise", "waise"},
            {"ende", "ente"},
            {"man", "mann"},
            {"hält", "hellt"},
            {"fälle", "felle"},
        )
    ],
    "en": [
        frozenset(s)
        for s in (
            {"there", "their", "theyre"},
            {"to", "too", "two"},
            {"hear", "here"},
            {"tale", "tail"},  # Alice ch. III: "Mine is a long and a sad tale!"
            {"knot", "not"},  # Alice ch. III: "I had not!" / "A knot!"
            {"weather", "whether"},
            {"right", "write", "rite"},
            {"your", "youre"},
            {"no", "know"},
            {"by", "buy", "bye"},
            {"sea", "see"},
            {"flour", "flower"},
            {"wait", "weight"},
            {"whole", "hole"},
            {"which", "witch"},
            {"one", "won"},
            {"new", "knew"},
            {"night", "knight"},
            {"tea", "tee"},
        )
    ],
}


def lookup_abbreviation(token: str, lang: Lang) -> str | None:
    """Return the spoken form of a written abbreviation, if known."""
    return ABBREVIATIONS[lang].get(token) or UNITS[lang].get(token)


def are_homophones(a: str, b: str, lang: Lang) -> bool:
    a, b = a.lower().replace("'", ""), b.lower().replace("'", "")
    return any(a in group and b in group for group in HOMOPHONES[lang])
