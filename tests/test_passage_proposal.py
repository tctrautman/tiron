from dataclasses import replace
import json
from pathlib import Path

from tiron.reconciliation import Word, reconcile_gap
from tiron.passage_proposal import propose_passage


def fixture():
    data = json.loads((Path(__file__).parent / 'data/p17/harris-introduction.json').read_text())
    return [Word(**w) for w in data['A']], [Word(**w) for w in data['B']], data['gap']


def test_recorded_passage_preserves_tokens_and_omits_internal_times():
    a, b, gap = fixture()
    result = propose_passage(a, b, gap, (6, 56))
    assert result['status'] == 'needs_review'
    assert result['passage']['tokens'] == [w.text for w in b[6:56]]
    assert result['passage']['word_timings'] is None
    assert result['passage']['start'] == b[6].start
    assert result['passage']['end'] == b[55].end
    assert result['passage']['speaker'] == 'A:3:4'
    assert [w.text for w in result['words']] == [w.text for w in a]
    assert reconcile_gap(a, b, gap)['status'] == 'abstained'


def test_weak_outer_word_is_not_promoted():
    a, b, gap = fixture()
    b[6] = replace(b[6], score=.01)
    assert 'weak_outer_boundary' in propose_passage(a, b, gap, (6, 56))['reasons']


def test_conflicting_speakers_remain_unresolved():
    a, b, gap = fixture()
    a[28] = replace(a[28], speaker='other')
    result = propose_passage(a, b, gap, (6, 56))
    assert result['passage']['speaker'] is None
    assert 'unresolved_speaker' in result['reasons']


def test_a_words_inside_envelope_cannot_be_deleted():
    a, b, gap = fixture()
    a.insert(34, replace(b[14], node='A:extra', speaker='other'))
    result = propose_passage(a, b, gap, (6, 56))
    assert result['status'] == 'abstained'
    assert result['words'] == a


def test_repeated_interior_words_are_kept():
    a, b, gap = fixture()
    b[15] = replace(b[15], text=b[14].text)
    result = propose_passage(a, b, gap, (6, 56))
    assert result['passage']['tokens'][8:10] == ['uh', 'uh']


def test_ambiguous_ownership_abstains():
    a, b, gap = fixture()
    a.insert(33, a[33])
    assert propose_passage(a, b, gap, (6, 56))['status'] == 'abstained'


def test_fallback_boundary_cannot_be_used():
    a, b, gap = fixture()
    b[55] = replace(b[55], timing='interpolated')
    assert propose_passage(a, b, gap, (6, 56))['status'] == 'abstained'


def test_unknown_anchor_identity_cannot_be_voted_away():
    a, b, gap = fixture()
    a[28] = replace(a[28], speaker=None)
    assert propose_passage(a, b, gap, (6, 56))['passage']['speaker'] is None


def test_clean_control_cannot_duplicate_existing_text():
    a, b, gap = fixture()
    assert propose_passage(a, b, (111, 113), (0, 6))['status'] == 'abstained'


def test_threshold_cannot_be_lowered():
    import pytest
    a, b, gap = fixture()
    with pytest.raises(ValueError):
        propose_passage(a, b, gap, (6, 56), min_score=.1)


def test_unmatched_adjacent_word_blocks_cut():
    a, b, gap = fixture()
    b[5] = replace(b[5], text='misrecognized')
    assert 'unanchored_seam' in propose_passage(a, b, gap, (6, 56))['reasons']
