# import torch
import librosa
import os
import onnxruntime as rt
from scipy.special import log_softmax
from scipy.io.wavfile import write as write_wav
from kfa.utils import get_trellis, merge_repeats, merge_words, backtrack

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

text = "|ɓɑn.toap.pii|prɑ.muk|riec|roat.thaa.phiʔ.ɓaal|kam.puʔ.cie|sɑm.ɗac|ʔaʔ.keaʔ|mɔ.haa|see.naa.paʔ.ɗəj|ɗee.coo|hun|saen|nɨw|tŋaj.tii|mphej|ɓəj|khaekum.pheaʔ|cnam|pii|poan|mphej|muəj|nih|ɓaan|sɑm.rac|ptɑl|ʔɑm.naoj|maah|krɑ.nat|cɑm.nuən|muəj|lien|cuun|roat.thaʔ.ɓaal|riec.thie.nii|pnum.pɨɲ|ɗaəm.ɓəj|caek.cuun|ɗɑl|ɓɑɑŋ.pʔoon|prɑ.cie.peaʔ..roat|nɨw|riec.thie.nii|pnum.pɨɲ|ɗaoj|ʔət|kɨt|tlaj|"
tokens = [vocabs.index(c) for c in text]
blank_id = vocabs.index("[PAD]")
y, sr = librosa.load("audio4-16khz.wav", sr=16000, mono=True)
model = rt.InferenceSession("wav2vec2-km-base-1500.onnx")
emissions = model.run(None, {"input": [y]})[0]
emissions = log_softmax(emissions, axis=-1)
emission = emissions[0]

trellis = get_trellis(emission, tokens, blank_id=blank_id)
path = backtrack(trellis, emission, tokens, blank_id=blank_id)
segments = merge_repeats(path, text)
word_segments = merge_words(segments)

os.makedirs("audio", exist_ok=True)
for i, word in enumerate(word_segments):
    ratio = y.shape[-1] / trellis.shape[0]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    segment = y[x0:x1]
    write_wav(f"audio/{i}-{word.label}.wav", sr, segment)
