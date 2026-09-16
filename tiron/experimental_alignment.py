"""Strict word alignment for offline reconciliation experiments.

Unlike display alignment this module never interpolates missing timings or
silently drops words. A forced path is timing evidence, not proof text was
spoken. Independent decoding and reviewed references remain necessary.
"""
import math

from .reconciliation import Word, key


def prepare_words(text, vocabulary):
    words = text.split()
    if not words:
        raise ValueError('empty text')
    normalized = [key(word) for word in words]
    if any(not token or any(c.isdigit() for c in word)
           for word, token in zip(words, normalized)):
        raise ValueError('unrepresentable word; no interpolation permitted')
    labels = '|'.join(normalized)
    if any(char not in vocabulary for char in labels):
        raise ValueError('unsupported aligner character')
    return words, [vocabulary[c] for c in labels]


def words_from_path(words, targets, path, scores, *, blank, delimiter,
                    frame_seconds, offset, node, speaker=None):
    if len(path) != len(scores) or frame_seconds <= 0:
        raise ValueError('invalid alignment dimensions')
    collapsed, previous, groups, frames = [], None, [], []
    for index, (token, score) in enumerate(zip(path, scores)):
        if not math.isfinite(score) or score > 0:
            raise ValueError('invalid aligned log probability')
        if token != blank and token != previous:
            collapsed.append(token)
        previous = token
        if token == delimiter:
            if frames:
                groups.append(frames)
                frames = []
        elif token != blank:
            frames.append(index)
    if frames:
        groups.append(frames)
    if collapsed != targets or len(groups) != len(words):
        raise ValueError('incomplete forced alignment; no fallback permitted')
    return [Word(word, offset + frames[0] * frame_seconds,
                 offset + (frames[-1] + 1) * frame_seconds, node, speaker,
                 sum(math.exp(scores[i]) for i in frames) / len(frames))
            for word, frames in zip(words, groups)]


def align_phrase(samples, text, *, processor, model, sample_rate, offset, node, speaker=None):
    """Align one bounded phrase with a caller-owned pinned CTC model."""
    import torch
    from torchaudio.functional import forced_align

    if sample_rate != 16000 or not 0 < len(samples) <= 30 * sample_rate:
        raise ValueError('expected at most 30 seconds of 16 kHz audio')
    vocabulary = processor.tokenizer.get_vocab()
    words, targets = prepare_words(text, vocabulary)
    inputs = processor(samples, sampling_rate=sample_rate, return_tensors='pt')
    parameter = next(model.parameters())
    with torch.inference_mode():
        logits = model(inputs.input_values.to(parameter.device, dtype=parameter.dtype)).logits.float()
    log_probs = logits.log_softmax(dim=-1).cpu()
    blank = processor.tokenizer.pad_token_id
    path, scores = forced_align(log_probs, torch.tensor([targets]), blank=blank)
    aligned = words_from_path(words, targets, path[0].tolist(), scores[0].tolist(),
                              blank=blank, delimiter=vocabulary['|'],
                              frame_seconds=model.config.inputs_to_logits_ratio / sample_rate,
                              offset=offset, node=node, speaker=speaker)
    free_text = processor.batch_decode(logits.argmax(dim=-1).cpu())[0]
    return aligned, free_text
