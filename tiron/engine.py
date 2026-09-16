"""In-process transformers inference for Tiron."""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

from . import config
from .constraints import TironConstraintLogitsProcessor, check_vocab_ids
from .decode import decode_with_specials, is_silent_chunk, parse_concat_segments, window_serving_params


class TironEngine:
    """Multi-speaker ASR with cross-window speaker linking.

    max_speakers caps the number of global identities produced by the linker
    and, when `constrained_decoding` is enabled, also caps the grammar's
    per-window speaker-slot pool.

    `constrained_decoding=True` (default) applies the same token grammar
    production serving enforces at decode time (see `tiron.constraints`);
    disabling it falls back to plain greedy decoding.
    """

    def __init__(
        self,
        model_id=config.MODEL_ID,
        device=None,
        dtype=None,
        ecapa_device=None,
        batch_size=None,
        hf_token=None,
        constrained_decoding=True,
    ):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        self.model_id = model_id
        self.hf_token = hf_token
        self.constrained_decoding = bool(constrained_decoding)
        self.device = device or self._automatic_device(torch)
        self.dtype = self._resolve_dtype(torch, dtype)
        self.ecapa_device = ecapa_device or self.device
        self.batch_size = int(
            batch_size
            if batch_size is not None
            else {"cuda": 8, "mps": 4}.get(self.device.split(":")[0], 1)
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.processor = WhisperProcessor.from_pretrained(model_id, token=hf_token)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, token=hf_token, torch_dtype=self.dtype
        ).to(self.device).eval()
        self._configure_generation()
        self._resolve_token_ids(LANGUAGES)
        self.ecapa = EncoderClassifier.from_hparams(
            source=config.ECAPA_MODEL,
            run_opts={"device": self.ecapa_device},
        )

    @staticmethod
    def _automatic_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch, dtype):
        if dtype is None:
            return {
                "cuda": torch.bfloat16,
                "mps": torch.float16,
            }.get(self.device.split(":")[0], torch.float32)
        if isinstance(dtype, str):
            aliases = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            try:
                return aliases[dtype.lower()]
            except KeyError as exc:
                raise ValueError(f"unsupported dtype: {dtype}") from exc
        return dtype

    def _configure_generation(self) -> None:
        model_config = self.model.config
        model_config.forced_decoder_ids = None
        model_config.suppress_tokens = []
        model_config.begin_suppress_tokens = []
        generation = self.model.generation_config
        generation.forced_decoder_ids = None
        generation.language = None
        generation.task = None
        generation.suppress_tokens = None
        generation.begin_suppress_tokens = None
        if hasattr(generation, "no_timestamps_token_id"):
            delattr(generation, "no_timestamps_token_id")
        generation.no_speech_threshold = None

    def _token_id(self, token: str, *, required: bool = True):
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            if required:
                raise RuntimeError(f"tokenizer is missing required token {token!r}")
            return None
        return int(token_id)

    def _resolve_token_ids(self, languages) -> None:
        from transformers import WhisperConfig

        self.sot_id = self._token_id("<|startoftranscript|>")
        self.transcribe_id = self._token_id("<|transcribe|>")
        self.nospeech_id = self._token_id("<|nospeech|>")
        self.speaker_token_ids = {
            index: self._token_id(f"<|speaker{index}|>")
            for index in range(1, config.MAX_LOCAL_SPEAKERS + 1)
        }
        self.speaker_id_to_idx = {
            token_id: index for index, token_id in self.speaker_token_ids.items()
        }
        self.notimestamps_id = self._token_id("<|notimestamps|>")
        self.ts_begin_id = self.notimestamps_id + 1
        self.ts_end_id = self._token_id("<|30.00|>")
        self.eos_id = self.tokenizer.eos_token_id
        self.window_token_id = self._token_id("<|window|>", required=False)
        self.language_token_ids = {}
        for code in languages:
            token_id = self._token_id(f"<|{code}|>", required=False)
            if token_id is not None:
                self.language_token_ids[code] = token_id
        if not self.language_token_ids:
            raise RuntimeError("tokenizer exposes no Whisper language tokens")
        self.language_code_by_id = {
            token_id: code for code, token_id in self.language_token_ids.items()
        }
        self.model_context_sec = float(
            getattr(self.feature_extractor, "chunk_length", config.CHUNK_MAX_SEC)
        )
        self.max_decoder_length = int(
            getattr(
                self.model.config,
                "max_target_positions",
                WhisperConfig().max_target_positions,
            )
        )
        # Drift guard: the constraint grammar's default vocab-id constants
        # assume whisper-large-v3-turbo offsets. Fail loudly here rather
        # than silently masking the wrong logits rows at decode time.
        check_vocab_ids(
            NOSPEECH_TOKEN_ID=self.nospeech_id,
            NOTS_TOKEN_ID=self.notimestamps_id,
            TS_BEGIN_ID=self.ts_begin_id,
            TS_END_ID=self.ts_end_id,
            EOS_TOKEN_ID=self.eos_id,
        )

    def normalize_language(self, language: str | None) -> str | None:
        """Normalize a Whisper code or common language name."""
        from transformers.models.whisper.tokenization_whisper import (
            LANGUAGES,
            TO_LANGUAGE_CODE,
        )

        raw = (language or "").strip().lower()
        if not raw or raw == "auto":
            return None
        code = raw if raw in LANGUAGES else TO_LANGUAGE_CODE.get(raw)
        if code is None or code not in self.language_token_ids:
            raise ValueError(f"unsupported language: {language!r}")
        return code

    def resolve_language(self, language: str | None, arr=None) -> str | None:
        code = self.normalize_language(language)
        if code is not None or arr is None:
            return code
        return self.detect_language(arr)

    def _features(self, arrays):
        import numpy as np

        batch = [np.asarray(arr, dtype=np.float32) for arr in arrays]
        return self.feature_extractor(
            batch, sampling_rate=config.SR, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

    def _decode_generated(self, token_ids) -> str:
        return decode_with_specials(
            token_ids,
            tokenizer_decode=lambda ids: self.tokenizer.decode(
                ids, skip_special_tokens=True
            ),
            eos_token_id=self.tokenizer.eos_token_id,
            nospeech_token_id=self.nospeech_id,
            ts_begin_id=self.ts_begin_id,
            ts_end_id=self.ts_end_id,
            speaker_id_to_idx=self.speaker_id_to_idx,
            window_token_id=self.window_token_id,
            window_seconds=30.0,
            reset_offset_on_speaker=self.window_token_id is not None,
        )

    def _build_constraint_processor(self, max_speakers, prompt_len):
        target_mode, num_windows = window_serving_params(
            self.window_token_id, config.CHUNK_MAX_SEC
        )
        return TironConstraintLogitsProcessor(
            prompt_len=prompt_len,
            speaker_token_ids=self.speaker_token_ids,
            ts_begin_id=self.ts_begin_id,
            ts_end_id=self.ts_end_id,
            nots_token_id=self.notimestamps_id,
            nospeech_token_id=self.nospeech_id,
            eos_token_id=self.eos_id,
            window_token_id=self.window_token_id,
            num_windows=num_windows,
            max_speakers=max_speakers,
            target_mode=target_mode,
            # <|nospeech|> is part of the trained grammar: silent windows
            # must be able to open with it instead of a speaker tag.
            allow_initial_nospeech=True,
        )

    def _generate_batch(
        self,
        arrays,
        language_code: str,
        max_new_tokens: int | None = None,
        max_speakers: int | None = None,
    ) -> list[str]:
        import torch

        language_id = self.language_token_ids[language_code]
        prefix = [self.sot_id, language_id, self.transcribe_id]
        limit = int(max_new_tokens or 444)
        limit = min(limit, max(1, self.max_decoder_length - len(prefix) - 1))
        features = self._features(arrays)
        decoder_input_ids = torch.tensor(
            [prefix] * len(arrays), dtype=torch.long, device=self.device
        )
        generate_kwargs = {}
        if self.constrained_decoding:
            generate_kwargs["logits_processor"] = [
                self._build_constraint_processor(max_speakers, len(prefix))
            ]
        with torch.no_grad():
            output = self.model.generate(
                input_features=features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=limit,
                do_sample=False,
                num_beams=1,
                **generate_kwargs,
            )
        return [
            self._decode_generated(row[len(prefix):].detach().cpu().tolist())
            for row in output
        ]

    def generate_window(
        self, arr, duration_s, language_code, max_new_tokens=None, max_speakers=None
    ) -> str:
        """Decode one fixed-length log-mel window."""
        del duration_s
        return self._generate_batch(
            [arr],
            language_code,
            max_new_tokens=max_new_tokens,
            max_speakers=max_speakers,
        )[0]

    def detect_language(self, arr) -> str:
        """Detect language with one decoder step restricted to language tokens."""
        import torch

        features = self._features([arr])
        decoder = torch.tensor([[self.sot_id]], device=self.device)
        with torch.no_grad():
            logits = self.model(
                input_features=features, decoder_input_ids=decoder
            ).logits[0, -1]
        ids = torch.tensor(
            list(self.language_code_by_id), dtype=torch.long, device=logits.device
        )
        best_id = int(ids[torch.argmax(logits.index_select(0, ids))].item())
        return self.language_code_by_id[best_id]

    def decode_audio(self, audio):
        """Decode a path, encoded bytes, or a 16 kHz mono numpy array."""
        import numpy as np
        import soundfile as sf

        if isinstance(audio, np.ndarray):
            arr = np.asarray(audio, dtype=np.float32)
            sr = config.SR
        else:
            source = (
                str(Path(audio))
                if isinstance(audio, (str, os.PathLike))
                else io.BytesIO(audio)
            )
            try:
                arr, sr = sf.read(source, dtype="float32")
            except Exception:
                import librosa

                tmp_name = None
                try:
                    if isinstance(audio, (str, os.PathLike)):
                        arr, sr = librosa.load(str(audio), sr=None, mono=True)
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=".audio", delete=False
                        ) as handle:
                            handle.write(bytes(audio))
                            tmp_name = handle.name
                        arr, sr = librosa.load(tmp_name, sr=None, mono=True)
                finally:
                    if tmp_name:
                        try:
                            os.unlink(tmp_name)
                        except OSError:
                            pass
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.ndim != 1:
            raise ValueError("audio must be mono or have channels on the last axis")
        if sr != config.SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=config.SR)
        return np.asarray(arr, dtype=np.float32)

    def decode_and_parse_chunks(
        self, chunk_arrs, chunk_durations, *, language_code, max_speakers=None
    ):
        """Decode batches and group parsed segments by local speaker."""
        decoded_all: list[str] = []
        for start in range(0, len(chunk_arrs), self.batch_size):
            decoded_all.extend(
                self._generate_batch(
                    chunk_arrs[start:start + self.batch_size],
                    language_code,
                    max_speakers=max_speakers,
                )
            )
        grouped = []
        for decoded, duration in zip(decoded_all, chunk_durations):
            by_speaker: dict[int, list] = {}
            if not is_silent_chunk(decoded):
                for segment in parse_concat_segments(
                    decoded, chunk_duration=float(duration)
                ):
                    by_speaker.setdefault(
                        int(segment.get("speaker_idx") or 1), []
                    ).append(segment)
            grouped.append(by_speaker)
        return grouped, decoded_all

    def _embed(self, arr, start_s, end_s):
        from .pipeline import embed
        return embed(arr, start_s, end_s, self.ecapa)

    def transcribe(
        self,
        audio,
        language="auto",
        max_speakers=None,
        two_pass=None,
        low_mass_ecapa_merge=config.DEFAULT_LOW_MASS_ECAPA_MERGE,
        demote_weak_spine=config.DEFAULT_DEMOTE_WEAK_SPINE,
        ecapa_smoothing=False,
        low_mass_context_merge=False,
        mode="legacy",
        capture_diagnostics=False,
    ) -> dict:
        """Transcribe a meeting and link local speakers across windows."""
        import numpy as np
        from . import pipeline

        if mode != "legacy":
            raise ValueError(f"unsupported transcription mode: {mode}")
        started = time.time()
        explicit_language = self.normalize_language(language)
        arr = self.decode_audio(audio)
        original_duration = len(arr) / config.SR
        capture = None
        if capture_diagnostics:
            from .diagnostics import DecodeCapture
            capture = DecodeCapture(sample_rate=config.SR,
                                    onset_pad_sec=config.PAD_START_SEC,
                                    original_samples=len(arr))
        if original_duration > config.MAX_AUDIO_SECONDS:
            raise ValueError(
                f"audio is {original_duration:.1f}s; maximum is "
                f"{config.MAX_AUDIO_SECONDS}s"
            )
        if len(arr) == 0:
            result = {
                "duration": 0.0,
                "language": explicit_language or "auto",
                "speakers": [],
                "segments": [],
                "num_chunks": 0,
                "elapsed_s": round(time.time() - started, 2),
                "two_pass": None,
            }
            if capture is not None:
                capture.record_pass("A", [], [], [], [])
                result["diagnostics"] = capture.finish({}, {})
            return result

        arr = pipeline.apply_onset_pad(arr, config.PAD_START_SEC)
        duration = len(arr) / config.SR
        chunks = (
            [(0.0, duration)]
            if duration <= config.CHUNK_MAX_SEC
            else pipeline.fixed_window_chunks(
                arr, duration, window_sec=config.CHUNK_MAX_SEC
            )
        )
        print(f"[tiron] audio: {duration:.1f}s chunks={len(chunks)}", flush=True)
        chunk_arrs = [
            arr[int(start * config.SR):int(end * config.SR)]
            for start, end in chunks
        ]
        chunk_durations = [float(end - start) for start, end in chunks]
        target_samples = int(config.CHUNK_MAX_SEC * config.SR)
        for index, chunk in enumerate(chunk_arrs):
            if len(chunk) < target_samples:
                chunk_arrs[index] = np.concatenate([
                    chunk,
                    np.zeros(target_samples - len(chunk), dtype=chunk.dtype),
                ])

        cap = None
        if max_speakers is not None:
            try:
                requested_cap = int(max_speakers)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_speakers must be an integer") from exc
            if requested_cap < 1:
                raise ValueError("max_speakers must be at least 1")
            cap = min(requested_cap, config.MAX_GLOBAL_SPEAKERS)

        effective_language = explicit_language or self.detect_language(chunk_arrs[0])
        if explicit_language is None:
            print(
                f"[tiron] auto-detected language: {effective_language}", flush=True
            )
        chunk_segs, _decoded = self.decode_and_parse_chunks(
            chunk_arrs, chunk_durations, language_code=effective_language, max_speakers=cap
        )

        if capture is not None:
            capture.record_pass("A", chunks, chunk_arrs, chunk_segs, _decoded)

        def decode_second_pass(arrays, durations):
            parsed, decoded = self.decode_and_parse_chunks(
                arrays, durations, language_code=effective_language, max_speakers=cap
            )
            return parsed, {"raw_text": decoded} if capture is not None else {}

        use_two_pass = config.TWO_PASS_DEFAULT if two_pass is None else bool(two_pass)
        global_ids, _windows, two_pass_diag = pipeline.link_with_optional_two_pass(
            arr=arr,
            duration=duration,
            chunks=chunks,
            chunk_segs=chunk_segs,
            chunk_arrs=chunk_arrs,
            decode_and_parse=decode_second_pass,
            ecapa=self.ecapa,
            embed_fn=self._embed,
            use_two_pass=use_two_pass,
            window_sec=config.CHUNK_MAX_SEC,
            pass_b_window_sec=config.TWO_PASS_B_WINDOW_SEC,
            pass_b_first_window_sec=config.TWO_PASS_B_FIRST_WINDOW_SEC,
            min_calib_samples=config.TWO_PASS_MIN_SAME,
            cal_demote_mass_sec=config.TWO_PASS_DEMOTE_MASS_SEC,
            cal_demote_windows=config.TWO_PASS_DEMOTE_WINDOWS,
            max_k_cap=cap,
            low_mass_ecapa_merge=low_mass_ecapa_merge,
            demote_weak_spine=demote_weak_spine,
            demote_spine_min_mass_sec=config.DEMOTE_SPINE_MIN_MASS_SEC,
            demote_spine_min_windows=config.DEMOTE_SPINE_MIN_WINDOWS,
            pad_chunks_to_samples=target_samples,
            log_prefix="[tiron]",
            **({"capture_pass": capture.record_pass} if capture is not None else {}),
        )

        segments = []
        for chunk_index, by_speaker in enumerate(chunk_segs):
            chunk_start, _ = chunks[chunk_index]
            for local_speaker, local_segments in by_speaker.items():
                global_id = global_ids[(chunk_index, local_speaker)]
                for segment in local_segments:
                    segments.append({
                        "speaker": f"SPEAKER_{global_id:02d}",
                        "start": round(chunk_start + segment["start"], 2),
                        "end": round(chunk_start + segment["end"], 2),
                        "text": segment["text"],
                        "_global_id": global_id,
                    })
        segments.sort(key=lambda segment: segment["start"])
        if low_mass_context_merge:
            segments, _ = pipeline.merge_low_mass_context_turns(segments)
        if ecapa_smoothing:
            segments, _ = pipeline.smooth_short_bridge_turns(segments)

        seen = []
        for segment in segments:
            if segment["speaker"] not in seen:
                seen.append(segment["speaker"])
        remap = {
            old: f"SPEAKER_{index:02d}" for index, old in enumerate(seen)
        }
        for segment in segments:
            segment["speaker"] = remap[segment["speaker"]]
            segment.pop("_global_id", None)
        pipeline.shift_segments_to_original_timeline(
            segments, config.PAD_START_SEC
        )
        result = {
            "duration": round(original_duration, 2),
            "language": effective_language,
            "speakers": sorted({segment["speaker"] for segment in segments}),
            "segments": segments,
            "num_chunks": len(chunks),
            "elapsed_s": round(time.time() - started, 2),
            "two_pass": two_pass_diag,
        }
        if capture is not None:
            result["diagnostics"] = capture.finish(
                global_ids, remap,
                post_link_merges=bool(low_mass_context_merge or ecapa_smoothing),
            )
        return result
