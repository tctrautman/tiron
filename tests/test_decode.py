from tiron.decode import decode_with_specials, is_silent_chunk, parse_concat_segments


def test_parse_multiple_speakers():
    decoded = (
        "<|speaker1|><|0.20|> hello there<|1.40|>"
        "<|speaker2|><|1.50|> yes<|2.10|>"
    )
    assert parse_concat_segments(decoded, chunk_duration=3.0) == [
        {
            "start": 0.2,
            "end": 1.4,
            "text": "hello there",
            "speaker_idx": 1,
        },
        {"start": 1.5, "end": 2.1, "text": "yes", "speaker_idx": 2},
    ]


def test_parse_silence_missing_tail_and_clamping():
    assert parse_concat_segments("<|nospeech|>", chunk_duration=30.0) == []
    missing_tail = parse_concat_segments(
        "<|speaker3|><|4.00|> final words", chunk_duration=9.0
    )
    assert missing_tail == [
        {
            "start": 4.0,
            "end": 9.0,
            "text": "final words",
            "speaker_idx": 3,
        }
    ]
    clamped = parse_concat_segments(
        "<|speaker1|><|29.00|> late<|45.00|>", chunk_duration=30.0
    )
    assert clamped[0]["start"] == 29.0
    assert clamped[0]["end"] == 30.0


def test_decode_with_specials_and_window_offsets():
    words = {70: " hello", 71: " again"}
    decoded = decode_with_specials(
        [60, 100, 70, 150, 200, 110, 71, 120, 999],
        tokenizer_decode=lambda ids: "".join(words[token] for token in ids),
        eos_token_id=999,
        nospeech_token_id=998,
        ts_begin_id=100,
        ts_end_id=180,
        speaker_id_to_idx={60: 1},
        window_token_id=200,
        window_seconds=30.0,
        reset_offset_on_speaker=True,
    )
    assert decoded == (
        "<|speaker1|><|0.00|> hello<|1.00|>"
        "<|30.20|> again<|30.40|>"
    )


def test_silent_chunk_detection():
    assert is_silent_chunk("<|nospeech|>")
    assert is_silent_chunk("<|speaker1|><|0.00|><|0.00|>")
    assert not is_silent_chunk("<|speaker1|><|0.00|> hello<|1.00|>")
