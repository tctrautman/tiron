"""Grammar-constrained decoding for Tiron's multi-speaker token stream.

Ports the same decode-time constraint grammar Trelis' hosted serving
enforces to a `transformers.LogitsProcessor` usable with `model.generate()`.
Plain greedy decoding without this grammar loses ~5 points of cpWER versus
hosted serving, because nothing stops the model from emitting an
ungrammatical token stream (a bare EOS right after a speaker tag, a skipped
speaker slot, a degenerate repeat loop, ...).

The state machine is driven entirely off the tokens generated so far
(the suffix of `input_ids` after the fixed `[sot, lang, transcribe]`
prompt prefix) and masks `scores` in place, mirroring how the production
serving grammar masks logits:

  - step 0: force `<|speaker1|>` (or `<|nospeech|>` when opted in).
  - speaker tag -> forced opening timestamp (the very first timestamp of
    the row is additionally capped at `max_init_ts_index`, so the model
    can't be forced into an early filler utterance on long leading
    silence; later opening timestamps are unconstrained).
  - opening timestamp -> forced text (EOS and control tokens suppressed).
  - inside text -> continue text or close the segment with a timestamp
    (a log-prob tie-breaker forces the close when the model is hedging
    between "more text" and "close now" but the timestamp mass wins in
    aggregate — carried over from the original processor).
  - closing timestamp -> EOS or an allowed next speaker tag. Which
    speaker tags are legal depends on `target_mode` (see below).
  - `<|nospeech|>` is only legal as the very first generated token; once
    emitted, the next (and only) legal token is EOS.
  - `no_repeat_ngram_size` blocks any next token that would recreate an
    n-gram already present in the row (mirrors HF's own loop-breaker).
  - `max_speakers` caps the pool of usable speaker slots (grammar-level
    cap; independent of any higher-level speaker-linking cap).
  - `<|window|>` extended-context tiling (`num_windows` boundaries
    allowed) is supported for parity but is inert for the public model's
    native 30s chunking (`num_windows=0`).

Speaker-transition modes (`target_mode`):
  - "speaker_blocks" (production default): a speaker tag opens a block;
    same-speaker continuation is close-ts -> open-ts (no repeated speaker
    tag), and the next speaker tag may only be the next contiguous new
    slot — never a repeat or an earlier slot.
  - "interleaved_each_utterance": a speaker tag precedes every utterance,
    including same-speaker continuations. After a closing timestamp any
    previously-introduced speaker (including the current one) or the
    next new slot is legal.
  - "interleaved_on_change": like each-utterance, but same-speaker
    continuation is close-ts -> open-ts, so the current speaker tag is
    excluded from the next-speaker set (only a change of speaker, or a
    new slot, reopens with a tag).

Vocab-id constants below are whisper-large-v3-turbo offsets (the public
checkpoint's base) and are the module defaults; callers should still pass
tokenizer-resolved ids explicitly and treat any mismatch against these
defaults as a drift signal (see `check_vocab_ids`).
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor

# --- Vocab-id defaults (whisper-large-v3-turbo offsets) ---------------------
NOSPEECH_TOKEN_ID = 50363   # <|nospeech|>
NOTS_TOKEN_ID = 50364       # <|notimestamps|>
TS_BEGIN_ID = 50365         # <|0.00|>
TS_END_ID = 51865           # <|30.00|>
EOS_TOKEN_ID = 50257
MAX_INIT_TS_INDEX = 1500    # <=30.0s cap on the row's very first timestamp.
NO_REPEAT_NGRAM_SIZE = 15

SPEAKER_TOKEN_NAMES = tuple(f"<|speaker{i}|>" for i in range(1, 9))
SPEAKER_TOKEN_IDS = tuple(range(51866, 51874))
WINDOW_TOKEN_NAME = "<|window|>"
WINDOW_TOKEN_ID = 51874

TARGET_MODES = (
    "speaker_blocks",
    "interleaved_each_utterance",
    "interleaved_on_change",
)

NEG_INF = float("-inf")


def check_vocab_ids(**resolved: int | None) -> None:
    """Drift guard: raise if a tokenizer-resolved id disagrees with the
    hardcoded defaults above. Callers pass the ids they resolved from the
    live tokenizer keyed by the module constant name they correspond to,
    e.g. `check_vocab_ids(NOSPEECH_TOKEN_ID=123, EOS_TOKEN_ID=456)`.
    """
    module_globals = globals()
    for name, value in resolved.items():
        if value is None:
            continue
        expected = module_globals.get(name)
        if expected is None:
            raise RuntimeError(f"check_vocab_ids: unknown constant {name!r}")
        if int(value) != int(expected):
            raise RuntimeError(
                f"tokenizer-resolved {name}={value} does not match the "
                f"expected default {expected}; the model's vocab layout has "
                "drifted from what tiron.constraints assumes."
            )


def _ngrams_to_block(token_ids: list[int], n: int) -> set[int]:
    """Candidate next-token ids that would recreate an n-gram already
    present in `token_ids` (mirrors HF's `no_repeat_ngram_size`).
    """
    if len(token_ids) < n:
        return set()
    prefix = tuple(token_ids[-(n - 1):])
    blocked: set[int] = set()
    for i in range(len(token_ids) - n + 1):
        if tuple(token_ids[i:i + n - 1]) == prefix:
            blocked.add(token_ids[i + n - 1])
    return blocked


def _is_ts(tid: int, ts_begin_id: int, ts_end_id: int) -> bool:
    return ts_begin_id <= tid <= ts_end_id


def _windows_in_current_block(
    out_ids: list[int],
    window_id: int,
    speaker_token_ids: set[int],
) -> int:
    """Number of `<|window|>` boundaries emitted since the last speaker
    token — the per-block tile budget resets at every speaker tag.
    """
    count = 0
    for tid in reversed(out_ids):
        if tid == window_id:
            count += 1
        elif tid in speaker_token_ids:
            break
    return count


def _first_ts_per_slot(
    out_ids: list[int],
    speaker_token_ids_sorted: tuple[int, ...],
    ts_begin_id: int,
    ts_end_id: int,
) -> dict[int, int]:
    """{1-based slot index: first timestamp token id emitted in that slot}."""
    result: dict[int, int] = {}
    current_slot: int | None = None
    for t in out_ids:
        if t in speaker_token_ids_sorted:
            current_slot = speaker_token_ids_sorted.index(t) + 1
        elif _is_ts(t, ts_begin_id, ts_end_id) and current_slot is not None and current_slot not in result:
            result[current_slot] = t
    return result


def _force_only_text(row: torch.Tensor, eos_token_id: int) -> None:
    """Whisper's special/control tokens start at EOS_TOKEN_ID, so masking
    that suffix removes timestamps, speaker tags, language/task tokens,
    nospeech, and EOS in one operation.
    """
    row[eos_token_id:] = NEG_INF


def _force_only_timestamps(row: torch.Tensor, ts_begin_id: int, ts_end_id: int) -> None:
    row[:ts_begin_id] = NEG_INF
    row[ts_end_id + 1:] = NEG_INF


def _effective_speakers(
    speaker_token_ids_sorted: tuple[int, ...],
    max_speakers: int | None,
) -> tuple[int, ...]:
    if not max_speakers:
        return speaker_token_ids_sorted
    capped = max(1, min(int(max_speakers), len(speaker_token_ids_sorted)))
    return speaker_token_ids_sorted[:capped]


def _allowed_next_speaker_ids(
    out_ids: list[int],
    speaker_token_ids_sorted: tuple[int, ...],
    max_speakers: int | None = None,
) -> list[int]:
    """interleaved_each_utterance: any seen speaker, or the next new slot."""
    effective = _effective_speakers(speaker_token_ids_sorted, max_speakers)
    seen = [effective.index(tid) for tid in out_ids if tid in effective]
    if not seen:
        return [effective[0]]
    max_seen = max(seen)
    max_allowed = min(max_seen + 1, len(effective) - 1)
    return list(effective[:max_allowed + 1])


def _allowed_next_block_speaker_ids(
    out_ids: list[int],
    speaker_token_ids_sorted: tuple[int, ...],
    max_speakers: int | None = None,
) -> list[int]:
    """speaker_blocks: only the next contiguous new slot (empty once the
    (capped) speaker pool is exhausted, which forces EOS).
    """
    effective = _effective_speakers(speaker_token_ids_sorted, max_speakers)
    seen = [effective.index(tid) for tid in out_ids if tid in effective]
    if not seen:
        return [effective[0]]
    next_idx = max(seen) + 1
    if next_idx >= len(effective):
        return []
    return [effective[next_idx]]


def _allowed_next_on_change_speaker_ids(
    out_ids: list[int],
    speaker_token_ids_sorted: tuple[int, ...],
    max_speakers: int | None = None,
) -> list[int]:
    """interleaved_on_change: any previously seen speaker except the
    current one, or the next new slot.
    """
    effective = _effective_speakers(speaker_token_ids_sorted, max_speakers)
    seen = [effective.index(tid) for tid in out_ids if tid in effective]
    if not seen:
        return [effective[0]]
    current_idx = seen[-1]
    max_seen = max(seen)
    max_allowed = min(max_seen + 1, len(effective) - 1)
    return [
        effective[idx] for idx in range(max_allowed + 1) if idx != current_idx
    ]


def _next_speakers_for_mode(
    out_ids: list[int],
    target_mode: str,
    speaker_token_ids_sorted: tuple[int, ...],
    max_speakers: int | None,
) -> list[int]:
    if target_mode == "speaker_blocks":
        return _allowed_next_block_speaker_ids(
            out_ids, speaker_token_ids_sorted, max_speakers=max_speakers
        )
    if target_mode == "interleaved_on_change":
        return _allowed_next_on_change_speaker_ids(
            out_ids, speaker_token_ids_sorted, max_speakers=max_speakers
        )
    return _allowed_next_speaker_ids(
        out_ids, speaker_token_ids_sorted, max_speakers=max_speakers
    )


class TironConstraintLogitsProcessor(LogitsProcessor):
    """transformers `LogitsProcessor` enforcing Tiron's multi-speaker grammar.

    Intended for a single non-beam `model.generate(..., num_beams=1)` call.
    `input_ids` is the full `[batch, seq]` sequence including the
    `[sot, lang, transcribe]` prompt prefix; `scores` is `[batch, vocab]`
    logits for the next token. Per-row state is derived fresh from
    `input_ids` every step (no cross-call state is kept), so instances are
    safe to reuse across independent `generate()` calls as long as
    `prompt_len` and the batch's grammar config stay the same.
    """

    def __init__(
        self,
        *,
        prompt_len: int = 3,
        speaker_token_ids: dict[int, int] | tuple[int, ...] | None = None,
        ts_begin_id: int = TS_BEGIN_ID,
        ts_end_id: int = TS_END_ID,
        nots_token_id: int = NOTS_TOKEN_ID,
        nospeech_token_id: int = NOSPEECH_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
        window_token_id: int | None = None,
        num_windows: int = 0,
        min_windows: int = 0,
        max_speakers: int | None = None,
        target_mode: str = "speaker_blocks",
        allow_initial_nospeech: bool = False,
        enforce_monotonic_first_ts: bool = False,
        no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
        max_init_ts_index: int = MAX_INIT_TS_INDEX,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"unsupported target_mode={target_mode!r}")
        if speaker_token_ids is None:
            speaker_ids_sorted = tuple(sorted(SPEAKER_TOKEN_IDS))
        elif isinstance(speaker_token_ids, dict):
            # {1-based slot index: token id} as produced by TironEngine.
            speaker_ids_sorted = tuple(
                speaker_token_ids[idx] for idx in sorted(speaker_token_ids)
            )
        else:
            speaker_ids_sorted = tuple(sorted(speaker_token_ids))
        if len(speaker_ids_sorted) < 1:
            raise ValueError("at least one speaker token id is required")

        self.prompt_len = int(prompt_len)
        self.speaker_token_ids_sorted = speaker_ids_sorted
        self.speaker_token_ids = set(speaker_ids_sorted)
        self.speaker1_token_id = speaker_ids_sorted[0]
        self.ts_begin_id = int(ts_begin_id)
        self.ts_end_id = int(ts_end_id)
        self.nots_token_id = int(nots_token_id)
        self.nospeech_token_id = int(nospeech_token_id)
        self.eos_token_id = int(eos_token_id)
        self.window_token_id = window_token_id
        self.num_windows = int(num_windows)
        self.min_windows = int(min_windows)
        self.max_speakers = max_speakers
        self.target_mode = target_mode
        self.allow_initial_nospeech = bool(allow_initial_nospeech)
        self.enforce_monotonic_first_ts = bool(enforce_monotonic_first_ts)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.max_init_ts_index = int(max_init_ts_index)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            out_ids = input_ids[batch_idx, self.prompt_len:].tolist()
            self._apply_row(scores[batch_idx], out_ids)
        return scores

    def _apply_row(self, row: torch.Tensor, out_ids: list[int]) -> None:
        ts_begin_id, ts_end_id = self.ts_begin_id, self.ts_end_id
        eos_token_id = self.eos_token_id
        window_id = self.window_token_id
        num_windows = self.num_windows if window_id is not None else 0

        # 1) Always suppress <|notimestamps|>; suppress <|nospeech|> after
        #    step 0 (only legal as the very first generated token).
        row[self.nots_token_id] = NEG_INF
        nospeech_save = row[self.nospeech_token_id].clone()
        row[self.nospeech_token_id] = NEG_INF
        eos_save = row[eos_token_id].clone()

        n = len(out_ids)
        if n == 0:
            speaker1_save = row[self.speaker1_token_id].clone()
            allow_nospeech = self.allow_initial_nospeech
            ns_save = nospeech_save if allow_nospeech else None
            allow_window0 = (
                window_id is not None
                and num_windows > 0
                and self.target_mode != "speaker_blocks"
            )
            window0_save = row[window_id].clone() if allow_window0 else None
            row[:] = NEG_INF
            row[self.speaker1_token_id] = speaker1_save
            if ns_save is not None:
                row[self.nospeech_token_id] = ns_save
            if window0_save is not None:
                row[window_id] = window0_save
            return

        # `<|nospeech|>` rows are trained as `<|nospeech|><|endoftext|>`.
        if out_ids[-1] == self.nospeech_token_id:
            row[:] = NEG_INF
            row[eos_token_id] = eos_save
            return

        window_allowed = (
            window_id is not None
            and num_windows > 0
            and out_ids.count(window_id) < num_windows
        )
        if window_allowed and self.target_mode == "speaker_blocks":
            per_block_cap = max(1, num_windows // len(self.speaker_token_ids_sorted))
            window_allowed = (
                _windows_in_current_block(out_ids, window_id, self.speaker_token_ids)
                < per_block_cap
            )

        last = out_ids[-1]
        last_is_ts = _is_ts(last, ts_begin_id, ts_end_id)
        penult_is_ts = (n >= 2) and _is_ts(out_ids[-2], ts_begin_id, ts_end_id)
        penult_is_spk = (n >= 2) and out_ids[-2] in self.speaker_token_ids
        penult_is_window = (n >= 2) and window_id is not None and out_ids[-2] == window_id
        last_is_spk = last in self.speaker_token_ids

        allow_eos = True

        if last_is_spk:
            allow_window_after_spk = (
                self.target_mode == "speaker_blocks" and window_allowed
            )
            window_save = row[window_id].clone() if allow_window_after_spk else None
            is_first_ts = not any(_is_ts(t, ts_begin_id, ts_end_id) for t in out_ids)
            _force_only_timestamps(row, ts_begin_id, ts_end_id)
            if window_save is not None:
                row[window_id] = window_save
            if is_first_ts:
                last_allowed = ts_begin_id + self.max_init_ts_index
                row[last_allowed + 1: ts_end_id + 1] = NEG_INF
            elif self.enforce_monotonic_first_ts:
                current_slot_idx = self.speaker_token_ids_sorted.index(last) + 1
                first_ts_so_far = _first_ts_per_slot(
                    out_ids[:-1], self.speaker_token_ids_sorted, ts_begin_id, ts_end_id
                )
                if current_slot_idx not in first_ts_so_far and current_slot_idx > 1:
                    prior_first_ts = first_ts_so_far.get(current_slot_idx - 1)
                    if prior_first_ts is not None:
                        row[ts_begin_id:prior_first_ts] = NEG_INF
            allow_eos = False
        elif window_id is not None and last == window_id:
            window_save = row[window_id].clone() if window_allowed else None
            if self.target_mode == "speaker_blocks":
                _force_only_timestamps(row, ts_begin_id, ts_end_id)
            else:
                next_speakers = _next_speakers_for_mode(
                    out_ids, self.target_mode, self.speaker_token_ids_sorted,
                    self.max_speakers,
                )
                spk_saves = {tid: row[tid].clone() for tid in next_speakers}
                row[:] = NEG_INF
                for tid, saved in spk_saves.items():
                    row[tid] = saved
            if window_save is not None:
                row[window_id] = window_save
            allow_eos = False
        elif last_is_ts:
            if penult_is_spk or penult_is_window:
                _force_only_text(row, eos_token_id)
                allow_eos = False
            elif penult_is_ts:
                _force_only_text(row, eos_token_id)
                allow_eos = False
            else:
                if self.target_mode == "speaker_blocks":
                    next_speakers = _allowed_next_block_speaker_ids(
                        out_ids, self.speaker_token_ids_sorted,
                        max_speakers=self.max_speakers,
                    )
                    ts_saves = row[ts_begin_id:ts_end_id + 1].clone()
                elif self.target_mode == "interleaved_on_change":
                    next_speakers = _allowed_next_on_change_speaker_ids(
                        out_ids, self.speaker_token_ids_sorted,
                        max_speakers=self.max_speakers,
                    )
                    ts_saves = row[ts_begin_id:ts_end_id + 1].clone()
                else:
                    next_speakers = _allowed_next_speaker_ids(
                        out_ids, self.speaker_token_ids_sorted,
                        max_speakers=self.max_speakers,
                    )
                    ts_saves = None
                next_spk_saves = {tid: row[tid].clone() for tid in next_speakers}
                window_save = row[window_id].clone() if window_allowed else None
                row[:] = NEG_INF
                if ts_saves is not None:
                    row[ts_begin_id:ts_end_id + 1] = ts_saves
                for tid, saved in next_spk_saves.items():
                    row[tid] = saved
                if window_save is not None:
                    row[window_id] = window_save
        else:
            # Inside text: continue text or close with a timestamp. No
            # speaker/EOS/nospeech until a closing timestamp is emitted.
            row[eos_token_id:ts_begin_id] = NEG_INF
            row[ts_end_id + 1:] = NEG_INF
            allow_eos = False

            # Tie-breaker: if the total probability mass on real timestamps
            # exceeds the max probability of any single text token, force a
            # timestamp this step. Otherwise a tight top-1 gap between
            # "close now" and "more text" can let text win the argmax even
            # though timestamps hold the majority of the mass in aggregate.
            logprobs_now = torch.nn.functional.log_softmax(row.float(), dim=-1)
            ts_lp = logprobs_now[ts_begin_id:ts_end_id + 1].logsumexp(dim=-1)
            non_ts_lp = torch.cat([
                logprobs_now[:ts_begin_id],
                logprobs_now[ts_end_id + 1:],
            ])
            max_non_ts = non_ts_lp.max()
            if ts_lp > max_non_ts:
                row[:ts_begin_id] = NEG_INF
                row[ts_end_id + 1:] = NEG_INF

        # n-gram repetition blocking (mirrors HF's no_repeat_ngram_size).
        if self.no_repeat_ngram_size:
            blocked = _ngrams_to_block(out_ids, self.no_repeat_ngram_size)
            for tok_id in blocked:
                row[tok_id] = NEG_INF

        if allow_eos:
            min_win = self.min_windows
            if not (
                min_win
                and window_id is not None
                and out_ids.count(window_id) < min_win
            ):
                row[eos_token_id] = eos_save
