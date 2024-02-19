import os
import re
import pickle
from kfa.number_verbalize import number_translate2ascii, number_replacer
from sosap import Model
from khmernormalizer import normalize
from khmercut import tokenize

vocabs = {
    ".": 0,
    "a": 1,
    "c": 2,
    "e": 3,
    "f": 4,
    "g": 5,
    "h": 6,
    "i": 7,
    "j": 8,
    "k": 9,
    "l": 10,
    "m": 11,
    "n": 12,
    "o": 13,
    "p": 14,
    "r": 15,
    "s": 16,
    "t": 17,
    "u": 18,
    "w": 19,
    "z": 20,
    "\u014b": 21,
    "\u0251": 22,
    "\u0253": 23,
    "\u0254": 24,
    "\u0257": 25,
    "\u0259": 26,
    "\u025b": 27,
    "\u0268": 28,
    "\u0272": 29,
    "\u0294": 30,
    "|": 31,
    "[UNK]": 32,
    "[PAD]": 33,
}

with open(os.path.join(os.path.dirname(__file__), "lexicon.pkl"), "rb") as infile:
    _LEXICONS = pickle.load(infile)

_G2P_MODEL = Model(os.path.join(os.path.dirname(__file__), "g2p.fst"))
RE_GENERIC_NUMBER = re.compile(r"\d+\.?\d*")


def _phonemize(text: str):
    text = text.lower()
    text = number_translate2ascii(text)
    text = RE_GENERIC_NUMBER.sub(number_replacer, text)

    if "▁" in text:
        return [
            lat for subtoken in text.split("▁") for lat in _phonemize(subtoken) + ["."]
        ]

    text = re.sub(r"[^\u1780-\u17d2a-z]+", "", text)
    if len(text.strip()) == 0:
        return None
    if text in _LEXICONS:
        return _LEXICONS[text]
    return _G2P_MODEL.phoneticize(text)


def tokenize_phonemize(text):
    text = normalize(text, remove_zwsp=True)
    for token in tokenize(text):
        phonemic = _phonemize(token)
        if phonemic is None:
            yield (token, phonemic)
            continue
        lattices = re.sub(r"\.+", ".", "".join(phonemic))
        token_ids = [vocabs[l] for l in lattices]
        yield (token, lattices, token_ids)
