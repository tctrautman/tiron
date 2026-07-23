# Tiron model-card benchmark — reproduction scripts

Reproduce the pooled cpWER numbers on the [model card](https://huggingface.co/Trelis/tiron)
using the public [`Trelis/tiron-eval-meetings`](https://huggingface.co/datasets/Trelis/tiron-eval-meetings)
dataset (17 whole meetings, splits `ami` / `icsi` / `notsofar`).

## Usage

```bash
pip install -e ..                                            # the tiron package itself
pip install datasets jiwer scipy whisper-normalizer           # eval-only extras
```

```bash
python eval/run_eval.py                                       # full battery, all 17 meetings
python eval/run_eval.py --splits ami --limit 1                 # quick smoke test
```

`--model` (default `Trelis/tiron`), `--splits` (default `ami,icsi,notsofar`), and `--limit`
(cap meetings per split) are the flags you'll actually reach for; see `--help` for the rest
(device/dtype/language/hf-token).

## Scoring convention

- **cpWER** (concatenated permutation-invariant WER): concatenate each speaker's utterances
  into one string, then take the best hypothesis-to-reference speaker mapping (Hungarian
  assignment on the WER cost matrix) and score the whole meeting as one clip.
- **Pooled** per corpus = `sum(errors) / sum(reference words)` across that corpus's meetings
  (a micro-average, not a mean of per-meeting percentages) — this is what "AMI 34.68" etc.
  below means. **Macro** = unweighted mean of the three pooled corpus numbers.
- **Masking**: NOTSOFAR `<UNKNOWN/>` spans (annotator-unintelligible audio, optionally
  deletable) mask out overlapping *hypothesis words only* — the matching reference text is
  already stripped by normalization, so this stops good-faith transcriptions of that audio
  from being penalized as insertions. AMI/ICSI masked spans mark unlabeled/QC-excluded audio
  and drop the *whole* reference utterance and hypothesis segment that overlaps them, on both
  sides — see `scoring.score_meeting` for the exact (corpus-conditional) logic.
- Text normalization (`scoring.normalize_english_spoken`) folds case, digit/word numerals,
  en-GB/US spelling, and punctuation, but keeps contractions and fillers — verbatim-meeting
  convention, not `whisper-english`.

## Expected numbers (model card)

Pooled cpWER%, lower is better. Whole-meeting battery has run-to-run variance from
speaker-count (K) estimation flips — expect **±1 pp** on a re-run, not exact reproduction.

| corpus       | Tiron |
|--------------|------:|
| AMI          | 34.68 |
| ICSI         | 21.24 |
| NOTSOFAR-1   | 36.23 |
