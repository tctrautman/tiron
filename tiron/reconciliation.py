"""Offline A/B insertion experiment; not connected to production transcribe.

Only acoustically aligned whole words may cross a join. Existing A words are
never deleted. Conflicts abstain for the entire requested region. A proposed
result still requires independent coverage and human-reference evaluation.
"""
from dataclasses import dataclass, replace
import math
import re


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    node: str
    speaker: str | None
    score: float
    timing: str = 'ctc_forced'


def key(text):
    return re.sub(r"[^A-Z']", '', text.upper())


def refine_shared_timings(a, b, gap, min_score, min_anchor_words):
    """Use uniquely co-timed common words from a fuller B phrase.

    Two strong distinct common words must first link the same A/B nodes.
    The 250 ms midpoint tolerance is experimental; ambiguous matches and any
    refinement that creates A/A overlap keep the original timing unchanged.
    No text or speaker labels change, and original A scores are retained.
    """
    pairs = {}
    for i, word in enumerate(a):
        candidates = [j for j, other in enumerate(b)
                      if key(word.text) == key(other.text)
                      and other.timing == 'ctc_forced' and other.score >= min_score
                      and abs((word.start + word.end - other.start - other.end) / 2) <= .25]
        if len(candidates) == 1:
            pairs[i] = candidates[0]
    pairs = {i: j for i, j in pairs.items() if list(pairs.values()).count(j) == 1}
    anchors = {}
    for i, j in pairs.items():
        if a[i].timing == 'ctc_forced' and a[i].score >= min_score:
            anchors.setdefault((a[i].node, b[j].node), set()).add(key(a[i].text))
    proposed, evidence = list(a), []
    for i, j in pairs.items():
        if (a[i].end >= gap[0] - .25 and a[i].start <= gap[1] + .25
                and a[i].timing == 'ctc_forced' and
                len(anchors.get((a[i].node, b[j].node), set())) >= min_anchor_words):
            proposed[i] = replace(a[i], start=b[j].start, end=b[j].end)
            if proposed[i] != a[i]:
                evidence.append(dict(a_index=i, evidence_node=b[j].node, b_index=j,
                                     before=[a[i].start, a[i].end], after=[b[j].start, b[j].end]))
    for item in evidence:
        i = item['a_index']
        for j, other in enumerate(proposed):
            if i == j:
                continue
            new_overlap = min(proposed[i].end, other.end) - max(proposed[i].start, other.start)
            old_overlap = min(a[i].end, a[j].end) - max(a[i].start, a[j].start)
            if new_overlap > max(0, old_overlap) + 1e-6:
                return a, []
            if (a[i].start - a[j].start) * (proposed[i].start - other.start) < 0:
                return a, []
    if (sorted(range(len(a)), key=lambda i: a[i].start) !=
            sorted(range(len(a)), key=lambda i: proposed[i].start)):
        return a, []
    return proposed, evidence


def reconcile_gap(a, b, gap, *, min_score=.5, min_anchor_words=2):
    """Return an auditable proposal, never a publishable success flag.

    Scores are mean aligned-token probabilities, not ASR confidence. The .5
    threshold is provisional. Speaker links require distinct, matching words
    with overlapping acoustic intervals and unanimous known A identities.
    Unknown A identities and conflicting anchors cannot be voted away.
    """
    start, end = gap
    if not all(math.isfinite(t) for t in gap) or end <= start:
        raise ValueError('invalid gap')
    if not 0 <= min_score <= 1 or min_anchor_words < 2:
        raise ValueError('invalid experimental thresholds')
    a, b = list(a), list(b)
    for word in a + b:
        if (not all(math.isfinite(t) for t in (word.start, word.end, word.score))
                or word.end <= word.start or not 0 <= word.score <= 1
                or not word.node or not key(word.text)):
            raise ValueError('invalid aligned word')
    original_a = a
    a, retimings = refine_shared_timings(a, b, gap, min_score, min_anchor_words)
    reasons, anchors, candidates = set(), {}, []
    for word in b:
        overlaps = [other for other in a if word.start < other.end and word.end > other.start]
        matches = [other for other in overlaps if key(other.text) == key(word.text)]
        ambiguous = len(matches) > 1 or any(
            sum(candidate.start < other.end and candidate.end > other.start
                and key(candidate.text) == key(other.text) for candidate in b) > 1
            for other in matches)
        if ambiguous:
            if word.start < end and word.end > start:
                reasons.add('ambiguous_word_match')
            continue
        trustworthy = (word.timing == 'ctc_forced' and word.score >= min_score)
        if (word.start < end and word.end > start and
                any(other.timing != 'ctc_forced' or other.score < min_score for other in overlaps)):
            reasons.add('weak_alignment')
        if matches and len(matches) == len(overlaps) and trustworthy:
            node_anchors = anchors.setdefault(word.node, [])
            node_anchors.extend((key(word.text), other.speaker) for other in matches
                                if other.timing == 'ctc_forced' and other.score >= min_score)
            # Existing A words own this acoustic occurrence, including when
            # the B phrase straddles a segment-level gap boundary.
            continue
        if word.start >= end or word.end <= start:
            continue
        if not trustworthy:
            reasons.add('weak_alignment')
        if overlaps:
            reasons.add('conflicting_overlap')
        candidates.append(word)
    candidates.sort(key=lambda word: (word.start, word.end, word.node))
    for previous, word in zip(candidates, candidates[1:]):
        if previous.end > word.start:
            reasons.add('competing_b_words')
    if reasons:
        return dict(status='abstained', words=original_a, insertions=[], reasons=sorted(reasons),
                    anchors=anchors, retimings=[],
                    review_candidates=candidates)
    insertions = []
    for word in candidates:
        evidence = anchors.get(word.node, [])
        speakers = {speaker for _, speaker in evidence}
        speaker = None
        if len({token for token, _ in evidence}) >= min_anchor_words and len(speakers) == 1:
            speaker = next(iter(speakers))
        if speaker is None:
            reasons.add('unresolved_speaker')
        insertions.append(replace(word, speaker=speaker))
    return dict(status='needs_review' if reasons else 'proposed',
                words=sorted(a + insertions, key=lambda word: word.start),
                insertions=insertions, reasons=sorted(reasons), anchors=anchors,
                retimings=retimings, review_candidates=[])
