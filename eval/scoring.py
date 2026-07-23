"""Self-contained cpWER scorer for whole-meeting multi-speaker transcripts.

Ported, behaviorally identical, from Trelis' internal evaluation harness —
the same code that produced the numbers on the Trelis/tiron model card
(text normalization, cpWER with Hungarian speaker mapping, and the
corpus-conditional masked-span rules). Validated to reproduce the reference
harness within floating-point noise on all 17 benchmark meetings.

No imports from that codebase — this file has no dependency beyond
``jiwer``, ``scipy``, and ``whisper-normalizer``, so it can run standalone
against the public ``Trelis/tiron-eval-meetings`` dataset.

Public API
----------
``score_meeting(ref_utts, hyp_segments, unknown_spans, corpus=None)`` scores
one whole meeting and returns ``(errs, ref_words)``.

``pooled(results)`` micro-averages a list of per-meeting results into
per-corpus (and macro) cpWER%.
"""

from __future__ import annotations

import re
from collections import defaultdict

# ------------------------------------------------------------------------- #
# 1. Text normalization (matches the reference implementation exactly)
# ------------------------------------------------------------------------- #

# Unicode apostrophe variants -> ASCII apostrophe. Vendored from
# the reference implementation.
_APOSTROPHE_VARIANTS = {
    "’": "'",  # right single quotation mark
    "‘": "'",  # left single quotation mark
    "ʼ": "'",  # modifier letter apostrophe
    "`": "'",  # grave accent
    "´": "'",  # acute accent
}


def normalize_apostrophes(text: str) -> str:
    """Normalize all apostrophe variants to the ASCII apostrophe."""
    for variant, replacement in _APOSTROPHE_VARIANTS.items():
        text = text.replace(variant, replacement)
    return text


_english_spoken_normalizer = None


def normalize_english_spoken(text: str) -> str:
    """English normalizer that folds *rendering only* (case, digit vs word
    numerals, en-GB/en-US spelling, thousands-commas, hyphenation,
    punctuation, bracketed non-speech annotations). Keeps contractions,
    fillers, and honorifics — the correct convention for verbatim meeting
    transcription. See the Studio source (module docstring above) for the
    full rationale; behavior here is byte-for-byte identical.

    Requires: pip install whisper-normalizer
    """
    if not text:
        return text or ""

    global _english_spoken_normalizer
    if _english_spoken_normalizer is None:
        from whisper_normalizer.english import (
            EnglishNumberNormalizer,
            EnglishSpellingNormalizer,
            remove_symbols_and_diacritics,
        )

        class _PatchedNumberNormalizer(EnglishNumberNormalizer):
            def postprocess(self, s: str) -> str:
                def combine_cents(m):
                    try:
                        return f"{m.group(1)}{m.group(2)}.{int(m.group(3)):02d}"
                    except ValueError:
                        return m.string

                def extract_cents(m):
                    try:
                        return f"¢{int(m.group(1))}"
                    except ValueError:
                        return m.string

                s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
                s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)
                # Deliberately omit: re.sub(r"\b1(s?)\b", r"one\1", s)
                # (upstream's cosmetic 1->one rule mangles "1,999" -> "one,999")
                return s

        number_norm = _PatchedNumberNormalizer()
        spelling_norm = EnglishSpellingNormalizer()
        _remove_symbols = remove_symbols_and_diacritics

        def _normalize(s: str) -> str:
            # 1. lowercase — speech carries no case.
            s = s.lower()
            # 1b. join single-letter hyphenated prefixes: e-mail -> email.
            s = re.sub(r"\b([a-z])-([a-z]+)\b", r"\1\2", s)
            # 2. normalize apostrophe variants.
            s = normalize_apostrophes(s)
            # 2b. digit-decade apostrophe: 1990's -> 1990s.
            s = re.sub(r"(\d)'s\b", r"\1s", s)
            # 3. drop content inside [...] / (...) — non-speech annotations.
            s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
            s = re.sub(r"\(([^)]+?)\)", "", s)
            # 4. tokenizer artifact: "team 's" -> "team's".
            s = re.sub(r"\s+'", "'", s)
            # 5. thousands-comma inside digit runs.
            s = re.sub(r"(\d),(\d)", r"\1\2", s)
            # 6. strip non-decimal periods.
            s = re.sub(r"\.([^0-9]|$)", r" \1", s)
            # 7. strip symbols/diacritics; keep currency/percent/apostrophe.
            s = _remove_symbols(s, keep=".%$¢€£'")
            # 8. number normalization (patched).
            s = number_norm(s)
            # 9. en-GB -> en-US spelling.
            s = spelling_norm(s)
            # 9b. okay -> ok.
            s = re.sub(r"\bokay\b", "ok", s)
            # 10. defensive cleanup of currency/percent that lost their digit anchor.
            s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
            s = re.sub(r"([^0-9])%", r"\1 ", s)
            # 11. collapse whitespace.
            return re.sub(r"\s+", " ", s).strip()

        _english_spoken_normalizer = _normalize

    return _english_spoken_normalizer(text)


# ------------------------------------------------------------------------- #
# 2. WER primitives (matches the reference implementation)
# ------------------------------------------------------------------------- #


def wer_abs(ref: str, hyp: str) -> tuple[int, int]:
    """Un-bounded WER: (edit_errors, ref_word_count). Used for cpWER, which
    aggregates absolute errors / absolute ref words (not row-bounded)."""
    from jiwer import wer as _wer

    r, h = normalize_english_spoken(ref), normalize_english_spoken(hyp)
    if not r and not h:
        return (0, 0)
    if not r:
        return (len(h.split()), 0)
    if not h:
        rt = len(r.split())
        return (rt, rt)
    rt = len(r.split())
    errs = int(round(_wer([r], [h]) * rt))
    return (errs, rt)


def cpwer_row(ref_slots: list[str], hyp_speakers: list[str]) -> tuple[int, int]:
    """Concatenated permutation-invariant WER for one clip.

    ref_slots / hyp_speakers: per-speaker concatenated text (0..K strings
    each). Returns total (errs, ref_words) under the best hyp-to-ref
    permutation (Hungarian assignment on the absolute-WER cost matrix).
    """
    from scipy.optimize import linear_sum_assignment

    n_refs = len(ref_slots)
    n_hyps = len(hyp_speakers)
    k = max(n_refs, n_hyps, 1)
    refs = ref_slots + [""] * (k - n_refs)
    hyps = hyp_speakers + [""] * (k - n_hyps)

    cost = [[0] * k for _ in range(k)]
    ref_lens = [0] * k
    for i, r in enumerate(refs):
        rt = len(normalize_english_spoken(r).split())
        ref_lens[i] = rt
        for j, h in enumerate(hyps):
            e, _ = wer_abs(r, h)
            cost[i][j] = e

    row_ind, col_ind = linear_sum_assignment(cost)
    total_err = sum(cost[i][j] for i, j in zip(row_ind, col_ind))
    total_ref = sum(ref_lens)
    return total_err, total_ref


# ------------------------------------------------------------------------- #
# 3. Span masking (two corpus-conditional mechanisms, matching the reference harness)
# ------------------------------------------------------------------------- #


def mask_hyp_words_in_spans(
    segments: list[dict], spans: list[tuple[float, float]]
) -> tuple[list[dict], int]:
    """Drop hypothesis WORDS whose (interpolated) time falls inside any span.

    Used for NOTSOFAR's `<UNKNOWN/>` (annotator-unintelligible) stretches:
    those are OPTIONALLY DELETABLE — text normalization already strips the
    tag from the reference, which would otherwise make silence there free
    while a good-faith transcription of the genuinely-spoken-but-
    unintelligible audio pays insertions. Hypothesis segments carry no
    per-word timing, so word times are linearly interpolated across the
    segment span (word midpoints); masking stays at word granularity so
    surrounding good words still score.

    Returns (masked_segments, n_words_dropped); segments left with no words
    are removed.
    """
    if not spans:
        return segments, 0
    out: list[dict] = []
    dropped = 0
    for seg in segments:
        try:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", s0))
        except (TypeError, ValueError):
            out.append(seg)
            continue
        words = str(seg.get("text") or "").split()
        if not words or s1 <= s0:
            if not any(s0 < e and s1 > s for s, e in spans) and words:
                out.append(seg)
            elif words:  # zero/negative duration inside a span: all-or-nothing
                dropped += len(words)
            else:
                out.append(seg)
            continue
        if not any(s0 < e and s1 > s for s, e in spans):
            out.append(seg)
            continue
        step = (s1 - s0) / len(words)
        kept = [
            w for i, w in enumerate(words)
            if not any(s <= s0 + (i + 0.5) * step <= e for s, e in spans)
        ]
        dropped += len(words) - len(kept)
        if kept:
            out.append({**seg, "text": " ".join(kept)})
    return out, dropped


def _drop_in_spans(items, get_start, get_end, spans):
    """Whole-item drop: remove any item overlapping any span."""
    if not spans:
        return items, 0
    kept = []
    dropped = 0
    for it in items:
        s, e = get_start(it), get_end(it)
        if any(s < b and e > a for a, b in spans):
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


# ------------------------------------------------------------------------- #
# 4. Corpus markup stripping (matches the reference implementation)
# ------------------------------------------------------------------------- #

_UNDERSCORE_ACRONYM_RE = re.compile(r"\b([A-Z](?:_[A-Z])+)(s?)_?\b")
_UPPER_PREFIX_UNDERSCORE_RE = re.compile(r"\b([A-Z])_(?=[A-Za-z])")
_TRAILING_SINGLE_LETTER_RE = re.compile(r"\b([A-Z])_(?=\W|$)")
_BRACKETED_META_NOISE_RE = re.compile(r"\[[^\]]+\]")
_STANDALONE_CORPUS_SYMBOL_RE = re.compile(r"(?<!\S)[$%#@]+(?!\S)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")


def strip_underscore_acronyms(text: str) -> str:
    """Collapse corpus spelled-letter markup: L_C_D_ -> LCD, O_ -> O."""
    if not text:
        return text
    text = _UNDERSCORE_ACRONYM_RE.sub(lambda m: m.group(1).replace("_", "") + m.group(2), text)
    text = _UPPER_PREFIX_UNDERSCORE_RE.sub(r"\1", text)
    return _TRAILING_SINGLE_LETTER_RE.sub(r"\1", text)


def strip_corpus_markup(text: str) -> str:
    """Normalize AMI/ICSI/NOTSOFAR orthographic conventions into product ASR
    text: spelled-letter underscores, standalone qualitative symbols, and
    bracketed meta-noise notes."""
    if not text:
        return text
    text = strip_underscore_acronyms(text)
    text = _BRACKETED_META_NOISE_RE.sub(" ", text)
    text = _STANDALONE_CORPUS_SYMBOL_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------- #
# 5. Whole-meeting grouping + scoring (matches the reference harness)
# ------------------------------------------------------------------------- #

# Corpora whose masked spans mark unlabeled/QC-excluded audio: the battery
# drops whole reference utterances AND whole hypothesis segments overlapping
# those spans (both sides scored only over annotated audio). NOTSOFAR's
# `<UNKNOWN/>` spans are the opposite convention (word-level hyp mask only,
# no ref drop — see mask_hyp_words_in_spans docstring) and is the default
# for any corpus not listed here.
_WHOLE_ITEM_DROP_CORPORA = {"ami", "icsi", "dipco"}


def _group_ref(utts: list[dict]) -> dict[str, str]:
    by_spk: dict[str, list[str]] = {}
    for utt in sorted(utts, key=lambda x: (x["begin_time"], x["end_time"])):
        spk = utt["speaker_id"]
        text = strip_corpus_markup(utt["text"])
        if text:
            by_spk.setdefault(spk, []).append(text)
    return {s: " ".join(t) for s, t in by_spk.items()}


def _group_hyp(segments: list[dict]) -> dict[str, str]:
    by_spk: dict[str, list[str]] = {}
    for seg in sorted(segments, key=lambda s: s["start"]):
        spk = seg["speaker"]
        text = strip_corpus_markup(seg.get("text", ""))
        if text:
            by_spk.setdefault(spk, []).append(text)
    return {s: " ".join(t) for s, t in by_spk.items()}


def score_meeting(
    ref_utts: list[dict],
    hyp_segments: list[dict],
    unknown_spans: list[list[float]] | None,
    corpus: str | None = None,
) -> tuple[int, int]:
    """Score one whole meeting: whole-meeting cpWER (Hungarian best-mapping
    concatenated WER across speakers).

    ref_utts: reference utterances, each ``{"speaker_id", "begin_time",
        "end_time", "text"}`` (seconds, meeting-absolute time).
    hyp_segments: harness output segments, each ``{"speaker", "start",
        "end", "text"}`` — i.e. ``TironEngine.transcribe(...)["segments"]``.
    unknown_spans: ``[[start, end], ...]`` masked-time spans for this
        meeting (absolute seconds), or None/[] if none.
    corpus: "ami" / "icsi" / "notsofar" / etc. — selects which masking
        convention applies (see ``_WHOLE_ITEM_DROP_CORPORA`` above). Unknown
        or unspecified corpora default to the NOTSOFAR convention (hyp
        word-level mask only), which is a no-op when ``unknown_spans`` is
        empty.

    Returns (errs, ref_words).
    """
    spans = [tuple(s) for s in (unknown_spans or [])]
    utts = ref_utts
    segs = hyp_segments

    if spans and (corpus or "").lower() in _WHOLE_ITEM_DROP_CORPORA:
        utts, _ = _drop_in_spans(utts, lambda u: u["begin_time"], lambda u: u["end_time"], spans)
        segs, _ = _drop_in_spans(segs, lambda s: s.get("start", 0.0), lambda s: s.get("end", 0.0), spans)
    elif spans:
        segs, _ = mask_hyp_words_in_spans(segs, spans)

    ref_by_spk = _group_ref(utts)
    hyp_by_spk = _group_hyp(segs)
    return cpwer_row(list(ref_by_spk.values()), list(hyp_by_spk.values()))


def pooled(results: list[dict]) -> dict:
    """Micro-average cpWER per corpus, plus an unweighted macro average.

    results: one dict per scored meeting, each with ``{"corpus", "errs",
        "ref_words"}`` (e.g. what ``score_meeting`` returns, tagged with the
        meeting's corpus).

    Returns ``{"<corpus>": {"errs", "ref_words", "cpwer_pct"}, ...,
        "macro": <float>}`` where "macro" is the unweighted mean of the
        per-corpus pooled cpWER%.
    """
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"errs": 0, "ref_words": 0})
    for r in results:
        a = agg[r["corpus"]]
        a["errs"] += r["errs"]
        a["ref_words"] += r["ref_words"]

    out: dict = {}
    for corpus, a in agg.items():
        pct = (100.0 * a["errs"] / a["ref_words"]) if a["ref_words"] else float("nan")
        out[corpus] = {"errs": a["errs"], "ref_words": a["ref_words"], "cpwer_pct": round(pct, 2)}
    if out:
        out["macro"] = round(sum(v["cpwer_pct"] for v in out.values()) / len(out), 2)
    return out
