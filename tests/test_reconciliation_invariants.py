"""Generated sequence checks; these are not additional real hearing trials."""
import random

from tiron.reconciliation import Word, reconcile_gap


def test_insertions_preserve_repetitions_and_existing_text_across_generated_cases():
    rng = random.Random(1731)
    for _ in range(100):
        tokens = [rng.choice(['the', 'budget', 'I', 'agree', 'yes']) for _ in range(24)]
        left = rng.randrange(3, 15)
        right = left + rng.randrange(1, 6)
        b = [Word(token, i * .3, i * .3 + .12, 'B:0:1', None, .95)
             for i, token in enumerate(tokens)]
        a = [Word(word.text, word.start + rng.uniform(-.01, .01),
                  word.end + rng.uniform(-.01, .01), 'A:0:1', 'speaker-a', .95)
             for i, word in enumerate(b) if not left <= i < right]
        before = list(a)
        result = reconcile_gap(a, b, (b[left].start - .02, b[right-1].end + .02))
        assert [word.text for word in result['words']] == tokens
        assert [word.text for word in result['insertions']] == tokens[left:right]
        assert [word.text for word in result['words'] if word.node.startswith('A')] == [word.text for word in a]
        assert a == before


def test_unmeasured_insertions_never_become_proposals():
    for timing in ['uniform', 'interpolated', 'unknown', '']:
        a = [Word('I', 0, .1, 'A:0:1', 'speaker-a', .99),
             Word('agree', .2, .4, 'A:0:1', 'speaker-a', .99)]
        b = [Word(word.text, word.start, word.end, 'B:0:1', None, .99) for word in a]
        b.append(Word('yes', .6, .8, 'B:0:1', None, .99, timing))
        result = reconcile_gap(a, b, (.4, 1))
        assert result['status'] == 'abstained'
        assert result['words'] == a
        assert not result['insertions']
