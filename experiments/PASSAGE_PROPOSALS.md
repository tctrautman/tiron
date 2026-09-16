# Offline passage proposal experiment

`tiron.passage_proposal.propose_passage(a, b, gap, (lo, hi))` produces a
review-only passage from the exact B token slice `[lo:hi]`. It is separate from
`reconcile_gap`; that API retains its strict word-level acceptance behavior.

The caller must nominate a contiguous B slice. This module does not select gaps,
prove that words were spoken, or assemble a publishable transcript.

## Acceptance

- Both outer words have measured CTC alignment at the existing 0.5 threshold
  or higher. The threshold cannot be reduced.
- The passage occupies one B speaker node and stays inside the requested region
  (250 ms boundary tolerance, at most 30 seconds).
- No A word or competing B word owns audio inside the proposed outer interval.
- At least one adjacent B context word uniquely matches A acoustically. Repeated
  ambiguous matches abstain. Existing shared-word refinement may repair a
  truncated A boundary, with its existing evidence and overlap checks recorded.
- A speaker is proposed only with at least two distinct strong matching words
  and unanimous known A identity. Unknown or conflicting identities stay unknown.
- Every successful proposal has `status="needs_review"`. Tokens retain original
  B order, punctuation, and repetition; joined text uses single spaces because
  the input Word objects do not retain original whitespace. A text and order
  remain unchanged. `word_timings=None` explicitly withholds *all* internal times.

## Observed case and limits

The recorded Harris introduction produces a passage spanning 112.97–126.59
seconds, with all 50 missing B tokens preserved. Weak internal scores affect
fillers **and** ordinary words (for example “I”, “that”, “are”), so this is not a
filler exception. The user's listening confirmation supports this one passage's
text; it is not an input to the algorithm or a general accuracy measurement.

This experiment separates evidence about outer boundaries from evidence about
internal timings. It does **not** improve the weak timing estimates or establish
that the proposed outer boundaries meet a human-measured timing tolerance.
Forced alignment cannot independently prove text correctness. Broader audio
controls, independent recognition, speaker continuity, and human timing checks
remain necessary before considering production use. No production callsite is
connected, and existing word-level reconciliation still abstains on this case.
