import math
import numpy as np
import librosa
import os
import onnxruntime as rt
from tqdm import tqdm
from scipy.special import log_softmax, softmax
from scipy.io.wavfile import write as write_wav
from kfa.utils import get_trellis, merge_repeats, merge_words, backtrack, time_to_frame

emission_interval = 30
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

text = "".join(
    [
        "|ɓɑn.toap.pii|prɑ.muk|riec|roat.thaa.phiʔ.ɓaal|kam.puʔ.cie|sɑm.ɗac|ʔaʔ.keaʔ|mɔ.haa|see.naa.paʔ.ɗəj|ɗee.coo|hun|saen|nɨw|tŋaj.tii|mphej|ɓəj|khaekum.pheaʔ|cnam|pii|poan|mphej|muəj|nih|ɓaan|sɑm.rac|ptɑl|ʔɑm.naoj|maah|krɑ.nat|cɑm.nuən|muəj|lien|cuun|roat.thaʔ.ɓaal|riec.thie.nii|pnum.pɨɲ|ɗaəm.ɓəj|caek.cuun|ɗɑl|ɓɑɑŋ.pʔoon|prɑ.cie.peaʔ..roat|nɨw|riec.thie.nii|pnum.pɨɲ|ɗaoj|ʔət|kɨt|tlaj|",
        "ɓɑn.toap.pii|prɑ.muk|riec|roat.thaa.phiʔ.ɓaal|kam.puʔ.cie|sɑm.ɗac|ʔaʔ.keaʔ|mɔ.haa|see.naa.paʔ.ɗəj|ɗee.coo|hun|saen|nɨw|tŋaj.tii|mphej|ɓəj|khaekum.pheaʔ|cnam|pii|poan|mphej|muəj|nih|ɓaan|sɑm.rac|ptɑl|ʔɑm.naoj|maah|krɑ.nat|cɑm.nuən|muəj|lien|cuun|roat.thaʔ.ɓaal|riec.thie.nii|pnum.pɨɲ|ɗaəm.ɓəj|caek.cuun|ɗɑl|ɓɑɑŋ.pʔoon|prɑ.cie.peaʔ..roat|nɨw|riec.thie.nii|pnum.pɨɲ|ɗaoj|ʔət|kɨt|tlaj|",
        "ɓɑn.toap.pii|prɑ.muk|riec|roat.thaa.phiʔ.ɓaal|kam.puʔ.cie|sɑm.ɗac|ʔaʔ.keaʔ|mɔ.haa|see.naa.paʔ.ɗəj|ɗee.coo|hun|saen|nɨw|tŋaj.tii|mphej|ɓəj|khaekum.pheaʔ|cnam|pii|poan|mphej|muəj|nih|ɓaan|sɑm.rac|ptɑl|ʔɑm.naoj|maah|krɑ.nat|cɑm.nuən|muəj|lien|cuun|roat.thaʔ.ɓaal|riec.thie.nii|pnum.pɨɲ|ɗaəm.ɓəj|caek.cuun|ɗɑl|ɓɑɑŋ.pʔoon|prɑ.cie.peaʔ..roat|nɨw|riec.thie.nii|pnum.pɨɲ|ɗaoj|ʔət|kɨt|tlaj|",
    ]
)

tokens = [vocabs.index(c) for c in text]
blank_id = vocabs.index("[PAD]")
y, sr = librosa.load("out.wav", sr=16000, mono=True)
total_duration = y.shape[-1] / sr
model = rt.InferenceSession("wav2vec2-km-base-1500.onnx")

i = 0
emissions_arr = []
with tqdm(total=math.ceil(total_duration / emission_interval)) as pbar:
    while i < total_duration:
        segment_start_time, segment_end_time = (i, i + emission_interval)
        context = emission_interval * 0.1
        input_start_time = max(segment_start_time - context, 0)
        input_end_time = min(segment_end_time + context, total_duration)
        y_chunk = y[int(sr * input_start_time) : int(sr * input_end_time)]
        emissions = model.run(None, {"input": [y_chunk]})[0]
        emissions = emissions[0]
        emission_start_frame = time_to_frame(segment_start_time)
        emission_end_frame = time_to_frame(segment_end_time)
        offset = time_to_frame(input_start_time)
        emissions = emissions[
            emission_start_frame - offset : emission_end_frame - offset, :
        ]
        emissions_arr.append(emissions)
        i += emission_interval
        pbar.update()

emissions = np.concatenate(emissions_arr, axis=0).squeeze()
stride = float(y.shape[-1] * 1000 / emissions.shape[0] / sr)
print(f"{stride=}")
emission = log_softmax(emissions, axis=-1)
trellis = get_trellis(emission, tokens, blank_id=blank_id)
path = backtrack(trellis, emission, tokens, blank_id=blank_id)
segments = merge_repeats(path, text)
word_segments = merge_words(segments)

os.makedirs("audio", exist_ok=True)

for i, word in enumerate(word_segments):
    ratio = y.shape[-1] / trellis.shape[0]
    x0 = (ratio * word.start / sr)
    x1 = (ratio * word.end / sr)
    print(word.label, (x0, x1))
    # segment = y[x0:x1]
    # write_wav(f"audio/{i}-{word.label}.wav", sr, segment)
