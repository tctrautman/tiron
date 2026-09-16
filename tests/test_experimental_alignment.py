import math
import pytest
from tiron.experimental_alignment import prepare_words, words_from_path


def test_unrepresentable_numeric_word_rejects_whole_phrase():
    with pytest.raises(ValueError, match='unrepresentable'):
        prepare_words('district 8', {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ'|")})


def test_measured_frames_preserve_original_tokens_and_path_scores():
    result = words_from_path(['I', 'am.'], [1, 3, 2, 4],
                             [0, 1, 1, 0, 3, 2, 4, 0], [math.log(.8)] * 8,
                             blank=0, delimiter=3, frame_seconds=.02,
                             offset=10, node='B:5:1')
    assert [w.text for w in result] == ['I', 'am.']
    assert result[0].start == 10.02 and result[0].end == 10.06
    assert result[1].start == 10.1 and result[1].end == 10.14
    assert result[1].score == pytest.approx(.8)


def test_missing_word_never_gets_uniform_fallback():
    with pytest.raises(ValueError, match='incomplete'):
        words_from_path(['I', 'am'], [1, 3, 2, 4], [0, 1, 0], [-.1] * 3,
                        blank=0, delimiter=3, frame_seconds=.02, offset=0, node='B:0:1')
