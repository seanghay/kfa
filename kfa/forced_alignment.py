import math
import numpy as np
import onnxruntime as rt
from tqdm import tqdm
from scipy.special import log_softmax
from kfa.utils import (
    get_trellis,
    merge_repeats,
    merge_words,
    backtrack,
    time_to_frame,
    intersperse,
    vocabs,
    download_file,
)
from kfa.text_normalize import tokenize_phonemize
import appdirs
import os
import shutil

_EMISSION_INTERVAL = 30

_MODEL_URL = "https://huggingface.co/seanghay/wav2vec2-base-khmer-phonetisaurus/resolve/main/wav2vec2-km-base-1500.onnx"
_MODEL_DIR = os.path.join(appdirs.user_cache_dir(), "kfa")
_MODEL_FILE_PATH = os.path.join(_MODEL_DIR, "wav2vec2-km-base-1500.onnx")
_MODEL_FILE_TMP_PATH = os.path.join(_MODEL_DIR, "wav2vec2-km-base-1500.onnx.tmp")

if not os.path.exists(_MODEL_FILE_PATH):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    download_file(_MODEL_URL, _MODEL_FILE_TMP_PATH)
    shutil.copy(_MODEL_FILE_TMP_PATH, _MODEL_FILE_PATH)


def create_session(providers=["CPUExecutionProvider"]):
    sess = rt.InferenceSession(_MODEL_FILE_PATH, providers=providers)
    return sess


def align(y, sr, text, session, silent=False):
    total_duration = y.shape[-1] / sr
    i = 0
    emissions_arr = []
    with tqdm(
        total=math.ceil(total_duration / _EMISSION_INTERVAL), disable=silent
    ) as pbar:
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + _EMISSION_INTERVAL)
            context = _EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            y_chunk = y[int(sr * input_start_time) : int(sr * input_end_time)]
            emissions = session.run(None, {"input": [y_chunk]})[0]
            emissions = emissions[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)
            emissions = emissions[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions)
            i += _EMISSION_INTERVAL
            pbar.update()

    emissions = np.concatenate(emissions_arr, axis=0).squeeze()
    emission = log_softmax(emissions, axis=-1)
    text_sequences = [
        c for l in text.split("\n") if l.strip() for c in tokenize_phonemize(l.strip())
    ]

    tokens = []
    texts = []
    original_tokens = []
    spans = []

    for i, item in enumerate(text_sequences):
        if len(item) == 2:
            spans[-1] += 1
            continue
        spans.append(0)
        original_tokens.append(item[0])
        tokens.append(item[2])
        texts.append(item[1])

    blank_id = vocabs["[PAD]"]
    text = "".join(intersperse(texts, "|"))
    tokens = [b for a in intersperse(tokens, [vocabs["|"]]) for b in a]
    trellis = get_trellis(emission, tokens, blank_id=blank_id)
    path = backtrack(trellis, emission, tokens, blank_id=blank_id)
    segments = merge_repeats(path, text)
    word_segments = merge_words(segments)
    second_start = 0
    for i, word in enumerate(word_segments):
        ratio = y.shape[-1] / trellis.shape[0]
        actual_second_start = ratio * word.start / sr

        second_end = ratio * word.end / sr
        actual_second_end = second_end

        if i < len(word_segments) - 1:
            second_end = max(ratio * word_segments[i + 1].start / sr, second_end)

        seq_idx = sum(spans[0:i]) + i
        span_size = spans[i]
        text_segment = "".join(
            map(lambda x: x[0], text_sequences[seq_idx : seq_idx + span_size + 1])
        )
        yield (
            text_segment,
            second_start,
            second_end,
            actual_second_start,
            actual_second_end,
        )
        second_start = second_end
