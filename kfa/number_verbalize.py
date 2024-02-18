import re
import math

DEFAULT_MINUS = "ដក"
DEFAULT_DELIMITER = "ក្បៀស"
DEFAULT_SEPARATOR = "▁"

digits = [
    "សូន្យ",
    "មួយ",
    "ពីរ",
    "បី",
    "បួន",
    "ប្រាំ",
    "ប្រាំមួយ",
    "ប្រាំពីរ",
    "ប្រាំបី",
    "ប្រាំបួន",
]

prefix = [
    "",
    "ដប់",
    "ម្ភៃ",
    "សាមសិប",
    "សែសិប",
    "ហាសិប",
    "ហុកសិប",
    "ចិតសិប",
    "ប៉ែតសិប",
    "កៅសិប",
]

suffix = dict(
    [
        (2, "រយ"),
        (3, "ពាន់"),
        (4, "ម៉ឺន"),
        (5, "សែន"),
        (6, "លាន"),
        (9, "ប៊ីលាន"),
        (12, "ទ្រីលាន"),
        (15, "ក្វាទ្រីលាន"),
        (18, "គ្វីនទីលាន"),
        (21, "សិចទីលាន"),
        (24, "សិបទីលាន"),
        (27, "អុកទីលាន"),
        (30, "ណូនីលាន"),
        (33, "ដេស៊ីលាន"),
        (36, "អាន់ដេស៊ីលាន"),
    ]
)


def integer(num, sep=DEFAULT_SEPARATOR, minus_sign=DEFAULT_MINUS):
    if math.isnan(num):
        return ""

    if num < 0:
        return minus_sign + sep + integer(abs(num), sep)

    num = math.floor(num)

    if num < 10:
        return digits[num]

    if num < 100:
        r = num % 10

        if r == 0:
            return prefix[num // 10]

        return prefix[num // 10] + integer(r, sep)

    exp = math.floor(math.log10(num))

    while exp not in suffix and exp > 0:
        exp = exp - 1

    d = 10**exp
    pre = integer(num // d, sep)
    suf = suffix[exp]
    r = num % d

    if r == 0:
        return pre + suf

    return pre + suf + sep + integer(r, sep)


def decimal(
    num, sep=DEFAULT_SEPARATOR, delimiter=DEFAULT_DELIMITER, minus_sign=DEFAULT_MINUS
):
    if math.isnan(num):
        return ""

    if isinstance(num, int):
        return integer(num, sep)

    right = str(num).split(".")[1]

    if len(right) > 3:
        word = sep.join([integer(int(c), sep) for c in right])
    else:
        word = integer(int(right), sep)

    n = math.trunc(num)
    prefix = minus_sign if n == 0 and num < 0 else ""
    return prefix + integer(n, sep, minus_sign=minus_sign) + sep + delimiter + word


KM_NUMBER_TABLE = str.maketrans(
    {
        "\u17e0": "0",
        "\u17e1": "1",
        "\u17e2": "2",
        "\u17e3": "3",
        "\u17e4": "4",
        "\u17e5": "5",
        "\u17e6": "6",
        "\u17e7": "7",
        "\u17e8": "8",
        "\u17e9": "9",
        ".": ",",
        ",": ".",
    }
)


def number_translate2ascii(text):
    return re.sub(
        r"([\u17e0-\u17e9]+[\,\.]?)+", lambda m: m[0].translate(KM_NUMBER_TABLE), text
    )


def number_verbalize(input_str):
    if input_str.isdigit():
        return integer(int(input_str))
    return decimal(float(input_str))

def number_replacer(m):    
    return number_verbalize(m[0])

