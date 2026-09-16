import json

import numpy as np
import pytest

from tiron import config, pipeline
from tiron.engine import TironEngine


def engine(monkeypatch, seconds=3):
    instance = object.__new__(TironEngine)
    instance.ecapa = None
    instance._embed = None
    instance.normalize_language = lambda value: 'en'
    instance.decode_audio = lambda value: np.ones(int(seconds * config.SR), dtype=np.float32)
    calls = []
    def decode(arrays, durations, **kwargs):
        calls.append(durations)
        return [{1: [{'start': .5, 'end': 1., 'text': 'hello'}]} for _ in arrays], ['raw'] * len(arrays)
    instance.decode_and_parse_chunks = decode
    monkeypatch.setattr(pipeline, 'link_speakers_global', lambda segs, *a, **k: ({(i, 1): 0 for i in range(len(segs))}, {}))
    return instance, calls


def test_unsupported_mode_rejected_before_audio(monkeypatch):
    instance, calls = engine(monkeypatch)
    instance.decode_audio = lambda _: pytest.fail('decoded audio before mode validation')
    with pytest.raises(ValueError, match='mode'):
        instance.transcribe(None, mode='contextual')
    assert not calls


def test_default_parity_and_json_capture(monkeypatch):
    instance, _ = engine(monkeypatch)
    plain = instance.transcribe(None)
    captured = instance.transcribe(None, mode='legacy', capture_diagnostics=True)
    diag = captured.pop('diagnostics')
    plain.pop('elapsed_s'); captured.pop('elapsed_s')
    assert plain == captured
    assert diag['pass_b_status'] == 'not_run'
    assert diag['passes'][0]['raw_text'][0] == 'raw'
    assert diag['speaker_mapping'][0]['final_speaker'] == 'SPEAKER_00'
    assert diag['sample_rate'] == config.SR
    json.dumps(diag, allow_nan=False)
    diag['passes'][0]['chunk_segs'][0].clear()
    assert instance.transcribe(None, capture_diagnostics=True)['diagnostics']['passes'][0]['chunk_segs'][0]


def test_b_capture_survives_calibration_failure(monkeypatch):
    instance, calls = engine(monkeypatch, 160)
    monkeypatch.setattr(pipeline, 'compute_node_embeddings', lambda *a: {})
    def fail(*a, **k):
        raise RuntimeError('calibration broke')
    monkeypatch.setattr(pipeline, 'link_speakers_calibrated', fail)
    result = instance.transcribe(None, capture_diagnostics=True, two_pass=True)
    assert len(calls) == 2
    assert result['two_pass']['mode'] == 'single_pass_fallback'
    b = result['diagnostics']['passes'][1]
    assert len(b['windows']) == len(calls[1])
    assert b['raw_text'][0] == 'raw'
    json.dumps(result, allow_nan=False)


def test_empty_audio_diagnostics(monkeypatch):
    instance, calls = engine(monkeypatch, 0)
    result = instance.transcribe(None, capture_diagnostics=True)
    assert result['diagnostics']['passes'][0]['windows'] == []
    assert not calls


def test_two_pass_success_preserves_output_and_native_windows(monkeypatch):
    instance, _ = engine(monkeypatch, 160)
    monkeypatch.setattr(pipeline, 'compute_node_embeddings', lambda *a: {})
    monkeypatch.setattr(pipeline, 'link_speakers_calibrated', lambda a, b, **k:
                        ({(i, 1): 0 for i in range(len(a['chunk_segs']))}, {}, {'engaged': True}))
    plain = instance.transcribe(None, two_pass=True)
    captured = instance.transcribe(None, two_pass=True, capture_diagnostics=True)
    diag = captured.pop('diagnostics')
    plain.pop('elapsed_s'); captured.pop('elapsed_s')
    assert captured == plain
    assert diag['original_samples'] == 160 * config.SR
    assert diag['onset_pad_samples'] == int(config.PAD_START_SEC * config.SR)
    assert diag['passes'][1]['windows'][0]['start_sample'] == 0
    # The 15-second target snaps to 12 on this constant-energy fixture.
    # Capture must retain the actual slice, not the nominal target.
    assert diag['passes'][1]['windows'][0]['end_sample'] == 12 * config.SR
    assert diag['passes'][1]['windows'][0]['input_samples'] == 30 * config.SR


def test_explicit_single_pass_does_not_decode_b(monkeypatch):
    instance, calls = engine(monkeypatch, 160)
    result = instance.transcribe(None, two_pass=False, capture_diagnostics=True)
    assert len(calls) == 1
    assert result['diagnostics']['pass_b_status'] == 'not_run'


def test_post_link_merge_does_not_claim_complete_node_mapping(monkeypatch):
    instance, _ = engine(monkeypatch)
    monkeypatch.setattr(pipeline, 'merge_low_mass_context_turns', lambda segments: (segments, {}))
    result = instance.transcribe(None, low_mass_context_merge=True, capture_diagnostics=True)
    assert result['diagnostics']['speaker_mapping_complete'] is False
    assert result['diagnostics']['speaker_mapping'] == []
