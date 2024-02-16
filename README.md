## KFA

A fast Khmer Forced Aligner powered by **Wav2Vec2CTC** and **Phonetisaurus**.

- [x] Built-in Speech Enhancement
- [x] Word-level Alignment

```shell
pip install kfa
```

#### CLI

```shell
kfa -i audio.wav -t text.txt -o alignments.jsonl
```

#### Python

```python
from kfa import align

align("audio.wav", "transcription")
```

#### References

- [MMS: Scaling Speech Technology to 1000+ languages](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [CTC FORCED ALIGNMENT API TUTORIAL](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
- [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)
- [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Thai Wav2vec2 model to ONNX model](https://pythainlp.github.io/tutorials/notebooks/thai_wav2vec2_onnx.html)


#### License

`Apache-2.0`
