import numpy as np
from dataclasses import dataclass
import requests
from tqdm import tqdm

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

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


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.shape[0] - 1, trellis.shape[1] - 1
    path = [Point(j, t, np.exp(emission[t, blank_id]))]
    while j > 0:
        assert t > 0

        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        t -= 1
        if changed > stayed:
            j -= 1

        prob = np.exp(p_change) if changed > stayed else np.exp(p_stay)
        path.append(Point(j, t, prob))

    while t > 0:
        prob = np.exp(emission[t - 1, blank_id])
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.shape[0]
    num_tokens = len(tokens)

    trellis = np.zeros((num_frame, num_tokens))
    trellis[1:, 0] = np.cumsum(emission[1:, blank_id], axis=0)
    trellis[0, 1:] = -np.inf
    trellis[-num_tokens + 1 :, 0] = np.inf

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = np.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
