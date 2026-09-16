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
