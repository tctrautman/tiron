"""Opt-in, per-call evidence for reproducing window-boundary omissions.

No audio, embeddings, tensors, or process-global capture buffers are retained.
Window samples refer to the padded input; segment times are local to a window.
Subtract onset_pad_samples/sample_rate to obtain the caller's audio timeline.
"""
from copy import deepcopy

UPSTREAM_BASE = "d249c5a81fc6e0f1ecd34fd30cf2519f06fe671c"
SCHEMA_VERSION = 1


class DecodeCapture:
    def __init__(self, *, sample_rate, onset_pad_sec, original_samples):
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "upstream_base": UPSTREAM_BASE,
            "mode": "legacy",
            "sample_rate": sample_rate,
            "original_samples": original_samples,
            "onset_pad_samples": int(onset_pad_sec * sample_rate),
            "coordinate_system": "padded_audio",
            "passes": [],
            "pass_b_status": "not_run",
            "speaker_mapping": [],
            "speaker_mapping_complete": True,
        }

    def record_pass(self, pass_id, chunks, chunk_arrs, chunk_segs, raw_text):
        if not (len(chunks) == len(chunk_arrs) == len(chunk_segs) == len(raw_text)):
            raise ValueError("decode capture cardinality mismatch")
        sr = self.data["sample_rate"]
        self.data["passes"].append({
            "pass_id": pass_id,
            "windows": [
                {"start_sample": int(start * sr), "end_sample": int(end * sr),
                 "input_samples": len(arr)}
                for (start, end), arr in zip(chunks, chunk_arrs)
            ],
            "chunk_segs": deepcopy(chunk_segs),
            "raw_text": list(raw_text),
        })
        if pass_id == "B":
            self.data["pass_b_status"] = "captured"

    def finish(self, global_ids, remap, *, post_link_merges=False):
        # Optional post-link turn rewrites can make node->speaker many-valued.
        # Never advertise a complete mapping when those rewrites were enabled.
        self.data["speaker_mapping_complete"] = not post_link_merges
        if not post_link_merges:
            self.data["speaker_mapping"] = [
                {"chunk_index": int(ci), "local_speaker": int(local),
                 "global_id": int(gid), "final_speaker": remap[f"SPEAKER_{gid:02d}"]}
                for (ci, local), gid in global_ids.items()
            ]
        return self.data
