"""Offline unit tests for the constraint-grammar LogitsProcessor.

No model, no network: these drive TironConstraintLogitsProcessor directly
with synthetic `input_ids` / `scores` tensors, mirroring how
`model.generate(..., logits_processor=[...])` calls it each step.
"""
from __future__ import annotations

import math

import torch

from tiron.constraints import (
    EOS_TOKEN_ID,
    MAX_INIT_TS_INDEX,
    NOSPEECH_TOKEN_ID,
    NOTS_TOKEN_ID,
    SPEAKER_TOKEN_IDS,
    TS_BEGIN_ID,
    TS_END_ID,
    TironConstraintLogitsProcessor,
)

VOCAB_SIZE = SPEAKER_TOKEN_IDS[-1] + 100
PROMPT_LEN = 3
SPEAKER1_ID, SPEAKER2_ID, SPEAKER3_ID = SPEAKER_TOKEN_IDS[:3]


def make_processor(**overrides) -> TironConstraintLogitsProcessor:
    kwargs = dict(prompt_len=PROMPT_LEN, target_mode="speaker_blocks")
    kwargs.update(overrides)
    return TironConstraintLogitsProcessor(**kwargs)


def run(processor, out_ids, vocab_size=VOCAB_SIZE):
    """Build a one-row batch, run the processor, and return the masked row."""
    input_ids = torch.tensor([[1, 2, 3] + list(out_ids)], dtype=torch.long)
    scores = torch.zeros((1, vocab_size), dtype=torch.float32)
    out = processor(input_ids, scores)
    return out[0]


def finite_ids(row: torch.Tensor) -> set[int]:
    return {i for i in range(row.shape[0]) if math.isfinite(row[i].item())}


def test_first_token_forces_speaker1():
    row = run(make_processor(), [])
    assert finite_ids(row) == {SPEAKER1_ID}


def test_first_token_allows_nospeech_when_enabled():
    row = run(make_processor(allow_initial_nospeech=True), [])
    assert finite_ids(row) == {SPEAKER1_ID, NOSPEECH_TOKEN_ID}


def test_speaker_token_forces_timestamp_and_caps_first_ts():
    row = run(make_processor(), [SPEAKER1_ID])
    allowed = finite_ids(row)
    assert allowed == set(range(TS_BEGIN_ID, TS_BEGIN_ID + MAX_INIT_TS_INDEX + 1))
    # notimestamps and EOS are never part of the allowed set.
    assert NOTS_TOKEN_ID not in allowed
    assert EOS_TOKEN_ID not in allowed


def test_second_timestamp_for_a_slot_is_not_capped():
    # speaker1 -> open ts -> text -> close ts -> speaker1 again (repeat is
    # not legal in speaker_blocks, but exercise the "not first ts" branch by
    # forcing a same-speaker re-open via interleaved_each_utterance mode.
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 10, 500, TS_BEGIN_ID + 20, SPEAKER1_ID]
    row = run(make_processor(target_mode="interleaved_each_utterance"), out_ids)
    allowed = finite_ids(row)
    # Uncapped: the full timestamp range is available past the first slot.
    assert allowed == set(range(TS_BEGIN_ID, TS_END_ID + 1))


def test_opening_timestamp_forces_text():
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5]
    row = run(make_processor(), out_ids)
    allowed = finite_ids(row)
    assert allowed == set(range(0, EOS_TOKEN_ID))


def test_closing_timestamp_allows_eos_and_next_contiguous_speaker():
    # speaker1 -> open ts -> text -> close ts: only speaker1 has appeared,
    # so the next legal speaker is speaker2 (next contiguous new slot),
    # plus EOS. speaker3 is not reachable yet.
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5, 500, TS_BEGIN_ID + 20]
    row = run(make_processor(), out_ids)
    allowed = finite_ids(row)
    assert SPEAKER2_ID in allowed
    assert SPEAKER1_ID not in allowed
    assert SPEAKER3_ID not in allowed
    assert EOS_TOKEN_ID in allowed


def test_speaker_cap_forces_eos_when_pool_exhausted():
    # With the speaker pool capped to 1, no new speaker slot can ever open.
    # speaker_blocks mode still allows further timestamps for the current
    # (only) speaker's block, plus EOS - but no speaker tag at all.
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5, 500, TS_BEGIN_ID + 20]
    row = run(make_processor(max_speakers=1), out_ids)
    allowed = finite_ids(row)
    assert allowed == {EOS_TOKEN_ID} | set(range(TS_BEGIN_ID, TS_END_ID + 1))
    assert not (allowed & set(SPEAKER_TOKEN_IDS))


def test_ngram_repetition_is_blocked():
    w1, w2 = 400, 401
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5, w1, w2, w1, w2]
    processor = make_processor(no_repeat_ngram_size=3)
    scores = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32)
    # Push all timestamp logits very low so the tie-breaker doesn't force a
    # close-ts on top of the n-gram block we're testing.
    scores[0, TS_BEGIN_ID:TS_END_ID + 1] = -100.0
    input_ids = torch.tensor([[1, 2, 3] + out_ids], dtype=torch.long)
    out = processor(input_ids, scores)[0]
    # The last two tokens are (w1, w2); that bigram already precedes w1 at
    # position 0, so re-emitting w1 next would recreate the trigram (w1, w2,
    # w1) that appears at positions [0:3] -> blocked.
    assert not math.isfinite(out[w1].item())
    # An unrelated text token stays legal.
    assert math.isfinite(out[402].item())


def test_nospeech_forces_eos_next():
    row = run(make_processor(), [NOSPEECH_TOKEN_ID])
    assert finite_ids(row) == {EOS_TOKEN_ID}


def test_eos_suppressed_immediately_after_speaker_token():
    row = run(make_processor(), [SPEAKER1_ID])
    assert not math.isfinite(row[EOS_TOKEN_ID].item())


def test_two_consecutive_timestamps_forces_text_recovery():
    out_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5, 500, TS_BEGIN_ID + 20, TS_BEGIN_ID + 25]
    row = run(make_processor(), out_ids)
    allowed = finite_ids(row)
    assert allowed == set(range(0, EOS_TOKEN_ID))


def test_batch_rows_are_independent():
    # Both rows have the same generated length (2 tokens), as a real batched
    # `generate()` step always does, but different histories -> different
    # legal next-token sets, proving rows are masked independently.
    processor = make_processor()
    row0_ids = [SPEAKER1_ID, TS_BEGIN_ID + 5]  # opening ts after speaker -> text only
    row1_ids = [500, TS_BEGIN_ID + 20]  # closing ts after text -> eos/speaker1
    input_ids = torch.tensor(
        [[1, 2, 3] + row0_ids, [1, 2, 3] + row1_ids], dtype=torch.long
    )
    scores = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32)
    out = processor(input_ids, scores)
    assert finite_ids(out[0]) == set(range(0, EOS_TOKEN_ID))
    assert finite_ids(out[1]) == (
        {SPEAKER1_ID, EOS_TOKEN_ID} | set(range(TS_BEGIN_ID, TS_END_ID + 1))
    )
