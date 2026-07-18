"""Dependency-light helpers for special-token decoding and segment parsing."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable


TOKEN_RE = re.compile(r"<\|(?:speaker\d+|nospeech|endoftext|\d+(?:\.\d+)?)\|>")
WORD_CHAR_RE = re.compile(r"\w", flags=re.UNICODE)


def decode_with_specials(
    token_ids: Iterable[int],
    *,
    tokenizer_decode: Callable[[list[int]], str],
    eos_token_id: int,
    nospeech_token_id: int,
    ts_begin_id: int,
    ts_end_id: int,
    speaker_id_to_idx: dict[int, int],
    window_token_id: int | None = None,
    window_seconds: float = 30.0,
    reset_offset_on_speaker: bool = False,
) -> str:
    """Decode text while preserving speaker, silence, and timestamp tokens."""
    out: list[str] = []
    buf: list[int] = []
    ts_offset = 0.0

    def flush() -> None:
        if buf:
            out.append(tokenizer_decode(buf))
            buf.clear()

    for raw_id in token_ids:
        token_id = int(raw_id)
        if token_id == eos_token_id:
            break
        if window_token_id is not None and token_id == window_token_id:
            flush()
            ts_offset += window_seconds
        elif ts_begin_id <= token_id <= ts_end_id:
            flush()
            out.append(f"<|{(token_id - ts_begin_id) * 0.02 + ts_offset:.2f}|>")
        elif token_id in speaker_id_to_idx:
            flush()
            if reset_offset_on_speaker:
                ts_offset = 0.0
            out.append(f"<|speaker{speaker_id_to_idx[token_id]}|>")
        elif token_id == nospeech_token_id:
            flush()
            out.append("<|nospeech|>")
        else:
            buf.append(token_id)
    flush()
    return "".join(out)


def is_silent_chunk(decoded: str) -> bool:
    """Return true for declared silence or an empty control-token-only body."""
    if "<|nospeech|>" in decoded:
        return True
    return not TOKEN_RE.sub("", decoded).strip()


def parse_concat_segments(
    decoded: str,
    *,
    chunk_duration: float | None = None,
) -> list[dict]:
    """Parse inline speaker and timestamp controls into local segments."""
    tokens: list[tuple[str, object]] = []
    last = 0
    for match in TOKEN_RE.finditer(decoded):
        if match.start() > last:
            text = decoded[last:match.start()]
            if text.strip():
                tokens.append(("text", text))
        tag = match.group(0)
        if tag.startswith("<|speaker"):
            tokens.append(("spk", int(tag[len("<|speaker"):-2])))
        elif tag == "<|nospeech|>":
            tokens.append(("nospeech", None))
        elif tag == "<|endoftext|>":
            tokens.append(("eos", None))
        else:
            tokens.append(("ts", float(tag[2:-2])))
        last = match.end()
    if last < len(decoded):
        tail = decoded[last:]
        if tail.strip():
            tokens.append(("text", tail))

    segments: list[dict] = []
    cur_start: float | None = None
    cur_speaker: int | None = None
    text_parts: list[str] = []
    last_end = 0.0

    def emit(end: float) -> None:
        nonlocal cur_start, last_end
        text = " ".join(text_parts).strip()
        if text and WORD_CHAR_RE.search(text) is not None:
            start = float(cur_start) if cur_start is not None else last_end
            segments.append({
                "start": start,
                "end": float(end),
                "text": text,
                "speaker_idx": cur_speaker if cur_speaker is not None else 1,
            })
            last_end = float(end)
        cur_start = None
        text_parts.clear()

    for kind, value in tokens:
        if kind == "ts":
            if text_parts:
                emit(float(value))
            else:
                cur_start = float(value)
        elif kind == "spk":
            if text_parts:
                emit(float(cur_start) if cur_start is not None else last_end)
            cur_speaker = int(value)
        elif kind == "nospeech":
            cur_start = None
            cur_speaker = None
            text_parts.clear()
        elif kind == "eos":
            break
        else:
            text_parts.append(str(value).strip())

    if text_parts:
        tail_end = (
            float(chunk_duration)
            if chunk_duration is not None
            else float(cur_start) if cur_start is not None else last_end
        )
        emit(tail_end)

    if chunk_duration is not None:
        for segment in segments:
            segment["start"] = max(0.0, min(segment["start"], chunk_duration))
            segment["end"] = max(
                segment["start"], min(segment["end"], chunk_duration)
            )
    return segments


def window_serving_params(
    window_token_id: int | None,
    chunk_seconds: float,
    *,
    window_seconds: float = 30.0,
    target_mode: str = "speaker_blocks",
) -> tuple[str, int]:
    """Return the block mode and maximum structural window boundaries."""
    if window_token_id is None:
        return "speaker_blocks", 0
    import math
    tiles_minus_one = max(
        0, math.ceil(float(chunk_seconds) / window_seconds) - 1
    )
    if target_mode == "interleaved_each_utterance":
        return target_mode, tiles_minus_one
    return "speaker_blocks", tiles_minus_one * 8


def speakers_text_in_first_emit_order(segments: list[dict]) -> list[str]:
    """Collect each local speaker's text in first-appearance order."""
    by_speaker: dict[int, list[str]] = {}
    seen: list[int] = []
    for segment in segments:
        index = int(segment["speaker_idx"])
        if index not in seen:
            seen.append(index)
        by_speaker.setdefault(index, []).append(str(segment["text"]))
    return [" ".join(by_speaker[index]).strip() for index in seen]
