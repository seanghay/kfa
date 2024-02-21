## KFA

[[Google Colab]](https://colab.research.google.com/drive/1-aRxWOzqqsL7Qbgp95dlwN-_cvI41INf?usp=sharing)

A fast Khmer Forced Aligner powered by **Wav2Vec2CTC** and **Phonetisaurus**.

- [ ] Built-in Speech Enhancement
- [x] Word-level Alignment

```shell
pip install kfa
```

#### CLI

```shell
kfa -a audio.wav -t text.txt -o alignments.jsonl
```

#### Python

```python
from kfa import align, create_session
import librosa

with open("test.txt") as infile:
    text = infile.read()
    
y, sr = librosa.load("text.wav", sr=16000, mono=True)
session = create_session()

for alignment in align(y, sr, text, session=session):
  print(alignment)
```

#### References

- [MMS: Scaling Speech Technology to 1000+ languages](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [CTC FORCED ALIGNMENT API TUTORIAL](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
- [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)
- [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Thai Wav2vec2 model to ONNX model](https://pythainlp.github.io/tutorials/notebooks/thai_wav2vec2_onnx.html)


#### License

`Apache-2.0`
