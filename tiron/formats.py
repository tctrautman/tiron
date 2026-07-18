"""Output renderers for JSON, WebVTT, SubRip, and plain text."""


def _fmt_timestamp_vtt(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:06.3f}"


def _fmt_timestamp_srt(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    sec, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"


def _escape_vtt(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_srt(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ")


def render(result: dict, fmt: str) -> tuple[str, str]:
    """Render a transcription result and return its body and media type."""
    if fmt == "json":
        import json
        return json.dumps(result, ensure_ascii=False), "application/json"
    segments = result["segments"]
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(
                f"{_fmt_timestamp_vtt(seg['start'])} --> "
                f"{_fmt_timestamp_vtt(seg['end'])}"
            )
            lines.append(
                f"<v {_escape_vtt(seg['speaker'])}>{_escape_vtt(seg['text'])}</v>"
            )
            lines.append("")
        return "\n".join(lines), "text/vtt"
    if fmt == "srt":
        lines = []
        for index, seg in enumerate(segments, 1):
            lines.extend([
                str(index),
                f"{_fmt_timestamp_srt(seg['start'])} --> "
                f"{_fmt_timestamp_srt(seg['end'])}",
                f"{_escape_srt(seg['speaker'])}: {_escape_srt(seg['text'])}",
                "",
            ])
        return "\n".join(lines), "application/x-subrip"
    if fmt == "text":
        lines, current = [], None
        for seg in segments:
            if seg["speaker"] != current:
                lines.append(f"\n[{seg['speaker']}]")
                current = seg["speaker"]
            lines.append(seg["text"])
        return "\n".join(lines).strip(), "text/plain"
    raise ValueError(f"unknown response format: {fmt}")
