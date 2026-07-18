import numpy as np

from tiron.config import SR
from tiron.pipeline import (
    apply_onset_pad,
    fixed_window_chunks,
    shift_segments_to_original_timeline,
)


def test_fixed_windows_cover_audio_without_gaps():
    rng = np.random.default_rng(4)
    duration = 95.0
    audio = rng.normal(0.0, 0.05, int(duration * SR)).astype(np.float32)
    chunks = fixed_window_chunks(audio, duration)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == duration
    assert all(0.0 < end - start <= 30.0 for start, end in chunks)
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))


def test_staggered_first_window_and_short_audio():
    audio = np.ones(int(70 * SR), dtype=np.float32)
    chunks = fixed_window_chunks(
        audio, 70.0, window_sec=25.0, first_window_sec=15.0
    )
    assert chunks[0][0] == 0.0
    assert chunks[0][1] <= 15.0
    assert all(end - start <= 25.0 for start, end in chunks)

    short = np.ones(int(2.25 * SR), dtype=np.float32)
    assert fixed_window_chunks(short, 2.25) == [(0.0, 2.25)]


def test_onset_pad_and_timeline_shift_round_trip():
    audio = np.ones(2 * SR, dtype=np.float32)
    padded = apply_onset_pad(audio, 0.75)
    assert len(padded) == len(audio) + int(0.75 * SR)
    assert np.max(np.abs(padded[: int(0.75 * SR)])) == 0.0
    segments = [
        {"start": 0.75, "end": 3.25, "text": "a"},
        {"start": 0.30, "end": 1.00, "text": "b"},
    ]
    shift_segments_to_original_timeline(segments, 0.75)
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 2.5
    assert segments[1]["start"] == 0.0 and segments[1]["end"] == 0.25
