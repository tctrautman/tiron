import numpy as np

from tiron.pipeline import link_speakers_calibrated, link_with_optional_two_pass


def _windows(duration, window, first=None):
    result, position = [], 0.0
    while position < duration:
        width = first if not result and first is not None else window
        end = min(position + width, duration)
        result.append((position, end))
        position = end
    return result


def _synthetic_pass(rng, windows, speaker_intervals, bases):
    chunk_segs, embeddings = [], {}
    for chunk_index, (start, end) in enumerate(windows):
        by_speaker = {}
        local = 0
        for speaker, intervals in enumerate(speaker_intervals):
            segments = []
            for seg_start, seg_end in intervals:
                clipped_start = max(seg_start, start)
                clipped_end = min(seg_end, end)
                if clipped_end - clipped_start > 0.05:
                    segments.append({
                        "start": clipped_start - start,
                        "end": clipped_end - start,
                        "text": "x",
                    })
            if segments:
                local += 1
                by_speaker[local] = segments
                vector = bases[speaker] + rng.normal(0, 0.01, bases[speaker].shape)
                vector = vector.astype(np.float32)
                embeddings[(chunk_index, local)] = {
                    "total_dur": sum(s["end"] - s["start"] for s in segments),
                    "is_clean": True,
                    "spine_emb": vector,
                    "concat_emb": vector,
                }
        chunk_segs.append(by_speaker)
    return {
        "chunk_windows": windows,
        "chunk_segs": chunk_segs,
        "emb": embeddings,
    }


def _meeting(duration=600.0):
    rng = np.random.default_rng(1)
    bases = [np.eye(1, 24, 0).ravel(), np.eye(1, 24, 1).ravel()]
    speaker_a = [
        (30 * index + 2.0, 30 * index + 10.0)
        for index in range(int(duration // 30))
    ]
    speaker_b = [
        (30 * index + 14.0, 30 * index + 26.0)
        for index in range(int(duration // 30))
    ]
    return (
        _synthetic_pass(
            rng, _windows(duration, 30.0), [speaker_a, speaker_b], bases
        ),
        _synthetic_pass(
            rng, _windows(duration, 25.0, 15.0), [speaker_a, speaker_b], bases
        ),
    )


def test_calibrated_linking_groups_two_voices():
    first, second = _meeting()
    identities, windows, diag = link_speakers_calibrated(
        first, second, min_calib_samples=30
    )
    assert diag["engaged"] is True
    assert len(set(identities.values())) == 2
    expected = {
        (chunk_index, speaker)
        for chunk_index, by_speaker in enumerate(first["chunk_segs"])
        for speaker in by_speaker
    }
    assert set(identities) == expected == set(windows)


def test_calibrated_linking_falls_back_when_under_evidenced():
    first, second = _meeting(duration=90.0)
    identities, windows, diag = link_speakers_calibrated(
        first, second, min_calib_samples=30
    )
    assert identities is None and windows is None
    assert diag["engaged"] is False


def test_single_chunk_passthrough_does_not_embed():
    audio = np.zeros(30, dtype=np.float32)

    def should_not_run(*_args):
        raise AssertionError("embedding should not run for one non-silent chunk")

    identities, windows, diag = link_with_optional_two_pass(
        arr=audio,
        duration=1.0,
        chunks=[(0.0, 1.0)],
        chunk_segs=[{1: [{"start": 0.0, "end": 1.0, "text": "hi"}]}],
        chunk_arrs=[audio],
        decode_and_parse=should_not_run,
        ecapa=None,
        embed_fn=should_not_run,
        use_two_pass=True,
    )
    assert identities == {(0, 1): 0}
    assert windows[(0, 1)]["attribution"] == "single_chunk_local"
    assert diag["mode"] == "single_pass"
