"""Offline passage review proposals. Never emits repaired Word objects."""
from .reconciliation import key, validate_alignment_inputs, refine_shared_timings


def propose_passage(a, b, gap, span, *, min_score=.5):
    """Propose the exact B token slice [lo:hi] with measured outer bounds.

    The caller nominates a contiguous passage, not individual strong words.
    Interior alignment is deliberately not exported. This cannot certify text
    or internal timing, and every successful result still needs review.
    """
    if not .5 <= min_score <= 1:
        raise ValueError('passage boundary threshold must be at least .5')
    a, b = validate_alignment_inputs(a, b, gap, min_score=min_score)
    lo, hi = span
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo < hi <= len(b):
        raise ValueError('invalid B token span')
    passage = b[lo:hi]
    first, last = passage[0], passage[-1]
    strong = lambda w: w.timing == 'ctc_forced' and w.score >= min_score
    reasons = set()
    if not strong(first) or not strong(last):
        reasons.add('weak_outer_boundary')
    if len({w.node for w in passage}) != 1:
        reasons.add('multiple_b_speakers')
    if not (gap[0] - .25 <= first.start < gap[1]
            and gap[0] < last.end <= gap[1] + .25
            and 0 < last.end - first.start <= 30):
        reasons.add('outside_requested_gap')
    if any(w.start < first.start or w.end > last.end for w in passage):
        reasons.add('invalid_passage_envelope')
    refined, retimings = refine_shared_timings(a, b, gap, min_score, 2)
    if any(w.start < last.end and w.end > first.start for w in refined):
        reasons.add('a_owns_passage_audio')
    if any(w.start < last.end and w.end > first.start for w in b[:lo] + b[hi:]):
        reasons.add('competing_b_audio')
    # All co-timed context matches participate, including unknown identities.
    anchors = []
    for j, word in enumerate(b):
        if lo <= j < hi or word.node != first.node or not strong(word):
            continue
        matches = [i for i, other in enumerate(a)
                   if key(other.text) == key(word.text)
                   and word.start < other.end and word.end > other.start]
        if len(matches) != 1:
            if matches:
                reasons.add('ambiguous_anchor')
            continue
        i = matches[0]
        if sum(key(other.text) == key(word.text) and other.start < a[i].end
               and other.end > a[i].start for other in b) != 1:
            reasons.add('ambiguous_anchor')
            continue
        if strong(a[i]):
            anchors.append((i, j))
    # Require a measured adjacent ownership seam; remote repeated phrases alone
    # cannot license cutting B at this index.
    adjacent = {lo - 1, hi}
    if not any(j in adjacent for _, j in anchors):
        reasons.add('unanchored_seam')
    speakers = {a[i].speaker for i, _ in anchors}
    speaker = None
    if len({key(a[i].text) for i, _ in anchors}) >= 2 and len(speakers) == 1:
        speaker = next(iter(speakers))
    if reasons:
        return dict(status='abstained', words=a, passage=None,
                    reasons=sorted(reasons), retimings=[])
    # Preserve A order exactly. The new passage has no synthetic Word entries.
    return dict(status='needs_review', words=refined,
                passage=dict(tokens=[w.text for w in passage],
                             text=' '.join(w.text for w in passage),
                             start=first.start, end=last.end, node=first.node,
                             speaker=speaker, word_timings=None,
                             source_span=[lo, hi], timing='outer_ctc_boundaries',
                             boundary_scores=[first.score, last.score]),
                reasons=['text_and_internal_timing_unverified'] +
                        (['unresolved_speaker'] if speaker is None else []),
                anchors=anchors, retimings=retimings)
