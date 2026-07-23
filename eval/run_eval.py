#!/usr/bin/env python3
"""Reproduce the Tiron model-card pooled cpWER numbers.

Loads the public ``Trelis/tiron-eval-meetings`` dataset (17 whole meetings
across the ``ami`` / ``icsi`` / ``notsofar`` splits), transcribes each
meeting with ``tiron.TironEngine``, scores it with ``eval/scoring.py``, and
prints a per-meeting table plus pooled per-corpus and macro cpWER%.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --model Trelis/tiron --splits ami,icsi --limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring  # noqa: E402

DATASET_ID = "Trelis/tiron-eval-meetings"
ALL_SPLITS = ("ami", "icsi", "notsofar")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Trelis/tiron", help="HF model repo or local checkpoint path")
    p.add_argument("--splits", default=",".join(ALL_SPLITS), help="comma-separated dataset splits to run")
    p.add_argument("--limit", type=int, default=None, help="max meetings per split (for a quick smoke run)")
    p.add_argument("--language", default="en", help="language passed to TironEngine.transcribe")
    p.add_argument("--device", default=None, help="cuda, mps, or cpu (auto-detected if omitted)")
    p.add_argument("--dtype", default=None, help="bf16, fp16, or fp32")
    p.add_argument("--hf-token", default=None, help="HF token (falls back to the HF_TOKEN env var / cached login)")
    return p.parse_args()


def _load_audio_for_engine(row: dict):
    """Return whatever `TironEngine.decode_audio` accepts: raw bytes, a
    file path, or an already-16kHz-mono float32 array.

    Handles both plausible HF `datasets` shapes for an ``audio`` column:
    already-decoded (``{"array", "sampling_rate"}`` via the `Audio`
    feature) or raw bytes (``{"bytes", "path"}`` with decoding disabled, or
    a plain ``bytes`` value).
    """
    audio = row["audio"]
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, str):
        return audio
    if isinstance(audio, dict):
        if audio.get("bytes"):
            return audio["bytes"]
        if "array" in audio:
            import numpy as np

            arr = np.asarray(audio["array"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sr = audio.get("sampling_rate") or 16000
            if sr != 16000:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        if audio.get("path"):
            return audio["path"]
    raise TypeError(f"unrecognized `audio` column value: {type(audio)!r}")


def _normalize_utt(u: dict) -> dict:
    """Accept small key-naming variance in `utterances_json` without
    requiring an exact schema match."""
    return {
        "speaker_id": u.get("speaker_id", u.get("speaker")),
        "begin_time": float(u.get("begin_time", u.get("start", u.get("start_time", 0.0)))),
        "end_time": float(u.get("end_time", u.get("end", u.get("end_time", 0.0)))),
        "text": u.get("text", ""),
    }


def main() -> int:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in ALL_SPLITS:
            print(f"error: unknown split {s!r}; expected one of {ALL_SPLITS}", file=sys.stderr)
            return 2

    from datasets import load_dataset
    from tiron import TironEngine

    print(f"[run_eval] loading model {args.model} ...", flush=True)
    engine = TironEngine(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        hf_token=args.hf_token,
    )

    per_meeting: list[dict] = []
    row_lines: list[str] = []

    for split in splits:
        print(f"[run_eval] loading split={split!r} of {DATASET_ID} ...", flush=True)
        ds = load_dataset(DATASET_ID, split=split, token=args.hf_token)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        for row in ds:
            meeting_id = row["meeting_id"]
            corpus = row.get("corpus") or split
            ref_utts = [_normalize_utt(u) for u in json.loads(row["utterances_json"])]
            unknown_spans = json.loads(row.get("unknown_spans_json") or "[]")
            audio = _load_audio_for_engine(row)

            t0 = time.time()
            result = engine.transcribe(audio, language=args.language)
            wall = time.time() - t0

            errs, ref_words = scoring.score_meeting(ref_utts, result["segments"], unknown_spans, corpus=corpus)
            pct = (100.0 * errs / ref_words) if ref_words else float("nan")
            per_meeting.append({"corpus": corpus, "errs": errs, "ref_words": ref_words})

            duration = row.get("duration_s")
            rtfx = (duration / wall) if duration and wall > 0 else float("nan")
            line = f"{meeting_id:<16} {corpus:<9} cpwer={pct:6.2f}%  errs={errs:5d}  ref_words={ref_words:5d}  rtfx={rtfx:5.1f}"
            print("[run_eval] " + line, flush=True)
            row_lines.append(line)

    print("\nPer-meeting results")
    print("-" * 70)
    for line in row_lines:
        print(line)

    pooled = scoring.pooled(per_meeting)
    print("\nPooled per-corpus cpWER% (micro-average: sum(errs) / sum(ref_words))")
    print("-" * 70)
    for corpus in ALL_SPLITS:
        if corpus in pooled:
            v = pooled[corpus]
            print(f"{corpus:<10} cpwer={v['cpwer_pct']:6.2f}%  errs={v['errs']:6d}  ref_words={v['ref_words']:6d}")
    if "macro" in pooled:
        print(f"{'macro':<10} cpwer={pooled['macro']:6.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
