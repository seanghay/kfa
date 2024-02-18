from kfa.number_verbalize import number_translate2ascii, number_replacer
import re
import pickle
from sosap import Model
from khmernormalizer import normalize
from khmercut import tokenize

vocabs = [
    ".",
    "a",
    "c",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "w",
    "z",
    "ŋ",
    "ɑ",
    "ɓ",
    "ɔ",
    "ɗ",
    "ə",
    "ɛ",
    "ɨ",
    "ɲ",
    "ʔ",
    "|",
    "[UNK]",
    "[PAD]",
]

with open("lexicon.pkl", "rb") as infile:
    _LEXICONS = pickle.load(infile)
_G2P_MODEL = Model("g2p.fst")

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


def read_lines(file):
    with open(file) as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            yield line


def _tokenize_phonemize(text):
    text = normalize(text, remove_zwsp=True)
    for token in tokenize(text):
        phonemic = _phonemize(token)
        if phonemic is None:
            yield (token, phonemic)
            continue
        lattices = re.sub(r"\.+", ".", "".join(phonemic))
        yield (token, [vocabs.index(l) for l in lattices])


if __name__ == "__main__":
    lines = [list(_tokenize_phonemize(l)) for l in read_lines("example.txt")]
    print(lines)
