import json
from pathlib import Path

from tiron.reconciliation import Word, reconcile_gap


def w(text, start, end, node='A:0:1', speaker='speaker-a', score=.9):
    return Word(text, start, end, node, speaker, score)


def test_crossing_phrase_recovers_only_missing_words():
    a = [w('I', 0, .2), w('represent', .3, .8), w('Next.', 3, 3.4, speaker='speaker-b')]
    b = [w('I', .01, .21, 'B:0:1', None), w('represent', .31, .81, 'B:0:1', None),
         w('North', 1, 1.4, 'B:0:1', None), w('Carolina.', 1.5, 2, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.8, 3))
    assert result['status'] == 'proposed'
    assert [x.text for x in result['words']] == ['I', 'represent', 'North', 'Carolina.', 'Next.']
    assert [x.speaker for x in result['insertions']] == ['speaker-a'] * 2
    assert a[1].text == 'represent' and b[2].speaker is None


def test_real_repetition_at_different_times_is_preserved():
    a = [w('I', 0, .2), w('agree', .3, .7)]
    b = [w('I', 0, .2, 'B:0:1', None), w('agree', .3, .7, 'B:0:1', None),
         w('agree', 1, 1.3, 'B:0:1', None)]
    assert [x.text for x in reconcile_gap(a, b, (.7, 2))['words']] == ['I', 'agree', 'agree']


def test_conflicting_word_at_join_abstains_without_deleting_a():
    a = [w('budget', .8, 1.2)]
    b = [w('budgets', .9, 1.3, 'B:0:1', None)]
    result = reconcile_gap(a, b, (1, 2))
    assert result['status'] == 'abstained'
    assert result['words'] == a
    assert 'conflicting_overlap' in result['reasons']


def test_b_only_speaker_is_explicitly_unresolved():
    result = reconcile_gap([], [w('Hello', 1, 2, 'B:0:2', None)], (0, 3))
    assert result['status'] == 'needs_review'
    assert result['insertions'][0].speaker is None
    assert 'unresolved_speaker' in result['reasons']


def test_conflicting_anchor_speakers_cannot_vote_away_ambiguity():
    a = [w('one', 0, .2), w('two', .3, .5, speaker='other')]
    b = [w('one', 0, .2, 'B:0:1', None), w('two', .3, .5, 'B:0:1', None),
         w('three', 1, 1.3, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.5, 2))
    assert result['status'] == 'needs_review'
    assert result['insertions'][0].speaker is None


def test_low_confidence_word_cannot_be_inserted():
    a = [w('before', 0, .5)]
    result = reconcile_gap(a, [w('uncertain', 1, 2, 'B:0:1', None, .01)], (.5, 3))
    assert result['status'] == 'abstained'
    assert result['words'] == a
    assert 'weak_alignment' in result['reasons']


def test_fallback_timing_cannot_supply_a_join():
    bad = Word('word', 1, 2, 'B:0:1', None, .9, timing='interpolated')
    assert reconcile_gap([], [bad], (0, 3))['status'] == 'abstained'


def test_clean_control_has_no_insertions():
    a = [w('complete', 0, .3), w('sentence', .5, 1)]
    b = [w('complete', 0, .3, 'B:0:1', None), w('sentence', .5, 1, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.3, .5))
    assert result['insertions'] == []
    assert result['words'] == a


def test_fuller_phrase_refines_truncated_end_without_changing_text():
    a = [w('detriment', 0, .4), w('to', .5, .6), w('the', .8, 1.1, score=.3)]
    b = [w('detriment', 0, .4, 'B:0:1', None), w('to', .5, .6, 'B:0:1', None),
         w('the', .65, .75, 'B:0:1', None), w('budget', .8, 1.1, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.8, 2))
    assert result['status'] == 'proposed'
    assert [word.text for word in result['words']] == ['detriment', 'to', 'the', 'budget']
    assert result['words'][2].end == .75
    assert result['retimings'][0]['evidence_node'] == 'B:0:1'


def test_multiple_nearby_same_words_do_not_authorize_retiming():
    a = [w('I', 0, .1), w('said', .15, .3), w('the', .5, .6)]
    b = [w('I', 0, .1, 'B:0:1', None), w('said', .15, .3, 'B:0:1', None),
         w('the', .4, .5, 'B:0:1', None), w('the', .6, .7, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.6, 1))
    assert result['retimings'] == []


def recorded_case(name):
    fixture = json.loads((Path(__file__).parent / 'data/p17' / (name + '.json')).read_text())
    return ([Word(**row) for row in fixture['A']],
            [Word(**row) for row in fixture['B']], fixture['gap'])


def test_recorded_employer_gap_is_joined_without_duplicate_the():
    # Acoustic-model regression evidence, not a human correctness reference.
    a, b, gap = recorded_case('harris-employer-coverage')
    result = reconcile_gap(a, b, gap)
    assert result['status'] == 'proposed'
    assert ' '.join(w.text for w in result['insertions']) == (
        'federal budget because of how expensive the Obamacare subsidies are.')
    assert {w.speaker for w in result['insertions']} == {'A:65:1'}
    assert [w.text for w in result['words'] if w.node.startswith('A')] == [
        w.text for w in sorted(a, key=lambda word: (word.start, word.end))]
    assert len(result['words']) == len(a) + 10


def test_recorded_introduction_needs_review_for_weak_words():
    a, b, gap = recorded_case('harris-introduction')
    result = reconcile_gap(a, b, gap)
    assert result['status'] == 'abstained'
    assert result['reasons'] == ['weak_alignment']
    assert result['words'] == a
    assert result['insertions'] == []
    assert any(w.text == 'uh' and w.score < .5 for w in result['review_candidates'])


def test_equal_start_a_words_keep_original_order():
    a = [w('first', 0, .8), w('second', 0, .3)]
    assert reconcile_gap(a, [], (1, 2))['words'] == a


def test_repetition_matching_multiple_a_words_abstains():
    a = [w('yes', 0, .5), w('yes', .2, .7)]
    b = [w('yes', .1, .6, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.4, 1))
    assert result['status'] == 'abstained'
    assert 'ambiguous_word_match' in result['reasons']
    assert result['words'] == a


def test_multiple_b_occurrences_matching_one_a_word_abstain():
    a = [w('yes', 0, .7)]
    b = [w('yes', .1, .3, 'B:0:1', None), w('yes', .4, .6, 'B:0:1', None)]
    result = reconcile_gap(a, b, (.2, 1))
    assert result['status'] == 'abstained'
    assert 'ambiguous_word_match' in result['reasons']
