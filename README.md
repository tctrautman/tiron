# Tiron

Reference inference harness for [Trelis/tiron](https://huggingface.co/Trelis/tiron), a multi-speaker meeting transcription model (Whisper large-v3 architecture with inline `<|speakerN|>` speaker tokens, up to 8 speakers).

The harness turns the model's 30-second windows into whole-meeting transcripts:

1. **Chunking** — fixed 30s windows with a 0.75s onset guardrail.
2. **Decoding** — greedy transformers decoding of each window into speaker-tagged, timestamped segments.
3. **Speaker linking** — ECAPA voice embeddings link window-local speakers into stable meeting-level identities, with a second staggered decode pass (on by default) that calibrates the clustering threshold per meeting.

Tiron is state of the art for whole-meeting transcription, leading every meeting test set we evaluated across open and proprietary systems. Its closest rival, AssemblyAI `universal-3-pro`, trails on every corpus (whole-meeting cpWER, same references and scoring, lower is better): AMI 33.31 vs 38.64, ICSI 21.19 vs 35.27, NOTSOFAR-1 34.84 vs 36.68. Details and meeting IDs on the [model card](https://huggingface.co/Trelis/tiron).

This harness uses the same grammar-constrained decoding as Trelis' hosted serving by default (disable with `--no-constrained-decoding` / `constrained_decoding=False`).

## Install

```bash
pip install -e .        # or: uv pip install -e .
```

Requires Python ≥3.10. CUDA, Apple Silicon (MPS), and CPU are supported (auto-detected).

## CLI

```bash
tiron meeting.wav                                  # JSON to stdout
tiron meeting.wav --format text                    # readable transcript
tiron meeting.wav --format srt --output out.srt    # subtitles (also: vtt)
tiron meeting.wav --language auto --max-speakers 4
```

## Python

```python
from tiron import TironEngine

engine = TironEngine()  # loads Trelis/tiron; pass hf_token=... if needed
result = engine.transcribe("meeting.wav", language="auto")

for seg in result["segments"]:
    print(f'[{seg["start"]:7.2f}-{seg["end"]:7.2f}] {seg["speaker"]}: {seg["text"]}')
```

`result` contains `segments` (each `{"speaker": "SPEAKER_00", "start", "end", "text"}` on the original file timeline), `speakers`, `language`, `duration`, and `two_pass` diagnostics.

## Tests

```bash
uv run --with pytest --with numpy pytest tests/ -q
```

## License

Apache 2.0.
