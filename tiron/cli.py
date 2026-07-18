"""Command-line interface."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe a meeting with Tiron")
    parser.add_argument("audio", help="path to an audio file")
    parser.add_argument("--language", default="auto", help="language code, name, or auto")
    parser.add_argument("--format", choices=("json", "vtt", "srt", "text"), default="json")
    parser.add_argument("--output", "-o", help="write output to this file")
    parser.add_argument("--model", default=None, help="checkpoint ID or local path")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--dtype", default=None, help="bf16, fp16, or fp32")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--no-two-pass", action="store_true")
    parser.add_argument(
        "--no-constrained-decoding",
        action="store_true",
        help="disable the token grammar and fall back to plain greedy decoding",
    )
    return parser


def main(argv=None) -> int:
    from . import config
    from .engine import TironEngine
    from .formats import render

    args = build_parser().parse_args(argv)
    engine = TironEngine(
        model_id=args.model or config.MODEL_ID,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        constrained_decoding=not args.no_constrained_decoding,
    )
    result = engine.transcribe(
        args.audio,
        language=args.language,
        max_speakers=args.max_speakers,
        two_pass=False if args.no_two_pass else None,
    )
    body, _media_type = render(result, args.format)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(body)
            if not body.endswith("\n"):
                handle.write("\n")
    else:
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
