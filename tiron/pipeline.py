"""Chunking, embedding, and cross-window speaker-linking utilities."""

from .config import (
    CHUNK_MAX_SEC,
    DEFAULT_DEMOTE_WEAK_SPINE,
    DEMOTE_SPINE_MIN_MASS_SEC,
    DEMOTE_SPINE_MIN_WINDOWS,
    MAX_GLOBAL_SPEAKERS,
    SR,
)

# Minimum clip length that produces a stable speaker embedding.
MIN_EMBED_SEC = 0.5
# Ignore tiny local fragments when deriving a within-window speaker floor.
MIN_DURABLE_SEC = 1.0
# Absorb clusters without enough clean speech to form a durable identity.
MIN_SPINE_MASS_SEC = 2.0
# Optional stricter evidence requirements for seeding global identities.
STRICT_SPINE_MIN_TOTAL_SEC = 3.0
STRICT_SPINE_MIN_EMBED_SEC = 1.0
# Require a clear nearest-centroid advantage for automatic attribution.
ATTRIBUTION_MARGIN = 1.3
# Reject automatic attribution to an acoustically distant centroid.
ATTRIBUTION_MAX_COSINE = 0.6
# Merge a small cluster only when its acoustic match is close and unambiguous.
LOW_MASS_MERGE_SEC = 10.0
LOW_MASS_MERGE_MAX_COSINE = 0.45
LOW_MASS_MERGE_MARGIN = 1.3
# Context merging is limited to short, strongly supported backchannels.
LOW_MASS_CONTEXT_SEC = 10.0
LOW_MASS_CONTEXT_MAX_SEG_SEC = 3.0
LOW_MASS_CONTEXT_MAX_WORDS = 8
LOW_MASS_CONTEXT_MIN_VOTES = 2
LOW_MASS_CONTEXT_MIN_DOMINANCE = 0.75
LOW_MASS_CONTEXT_MAX_GAP_SEC = 1.0
# Fixed cosine cut keeps speaker counts stable across small numeric changes.
CLUSTER_DISTANCE_THRESHOLD = 0.4


def _ecapa_device(ecapa):
    """Return the device configured on a SpeechBrain encoder."""
    import torch
    device = getattr(ecapa, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(ecapa.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _cosine(left, right) -> float:
    """Cosine distance with stable handling for zero vectors."""
    import numpy as np
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0 if np.array_equal(left, right) else 1.0
    return float(1.0 - np.dot(left, right) / denom)


def _linear_sum_assignment(cost):
    """Small-matrix exact assignment used for at most eight speaker slots."""
    import itertools
    import numpy as np
    matrix = np.asarray(cost)
    rows, cols = matrix.shape
    if rows > cols:
        trans_rows, trans_cols = _linear_sum_assignment(matrix.T)
        return trans_cols, trans_rows
    best_columns = None
    best_cost = float("inf")
    row_ids = tuple(range(rows))
    for columns in itertools.permutations(range(cols), rows):
        value = sum(float(matrix[row, col]) for row, col in zip(row_ids, columns))
        if value < best_cost:
            best_cost = value
            best_columns = columns
    return np.asarray(row_ids, dtype=int), np.asarray(best_columns, dtype=int)


def _agglomerative_labels(distance_matrix, *, n_clusters=None, distance_threshold=None):
    """Average-linkage agglomeration over a precomputed distance matrix."""
    import numpy as np
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    clusters = [{index} for index in range(len(matrix))]
    target = int(n_clusters) if n_clusters is not None else 1
    while len(clusters) > target:
        best = None
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                values = [
                    matrix[left, right]
                    for left in clusters[left_index]
                    for right in clusters[right_index]
                ]
                score = float(np.mean(values))
                candidate = (score, left_index, right_index)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        score, left_index, right_index = best
        if distance_threshold is not None and score > float(distance_threshold):
            break
        clusters[left_index] |= clusters[right_index]
        del clusters[right_index]
    labels = [0] * len(matrix)
    for label, members in enumerate(clusters):
        for member in members:
            labels[member] = label
    return labels


# --- onset guardrail ---
def apply_onset_pad(arr, pad_sec: float, sr: int = SR):
    """Prepend ``pad_sec`` of digital silence ONCE at the start of a full
    input file. A full-energy mid-word onset can make the model defer output
    to the next clean boundary. Apply this only at file start and pair it with
    :func:`shift_segments_to_original_timeline`.
    """
    if pad_sec <= 0:
        return arr
    import numpy as np
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([np.zeros(int(pad_sec * sr), dtype=np.float32), arr])


def shift_segments_to_original_timeline(segments, pad_sec: float):
    """Shift segment start/end times back by the onset pad (in place) so
    consumers see the original, unpadded file timeline. Clamped at 0."""
    if pad_sec <= 0:
        return segments
    for s in segments:
        s["start"] = round(max(0.0, float(s["start"]) - pad_sec), 2)
        s["end"] = round(max(0.0, float(s["end"]) - pad_sec), 2)
    return segments


# --- chunking ---
def fixed_window_chunks(
    audio, duration: float, window_sec: float = CHUNK_MAX_SEC,
    first_window_sec: float | None = None,
) -> list[tuple[float, float]]:
    """Chunks of ≤ window_sec, cut at energy-based silence boundaries
    where possible (else hard-cut at window_sec).

    Cutting mid-word at an exact fixed boundary removes useful acoustic
    context. By rolling forward and snapping each cut to the lowest-energy moment
    within a small "search window" near the window_sec boundary,
    chunks align with natural speech pauses while staying ≤ window_sec.

    ``window_sec`` defaults to ``CHUNK_MAX_SEC`` (30s).

    ``first_window_sec`` caps only the FIRST chunk (still energy-snapped).
    Passing window_sec/2 yields a staggered grid offset half a window from
    the default grid for the optional second-pass evidence. Default ``None``
    keeps the single-grid behavior.

    Energy detection is RMS over 50 ms frames — pure numpy, no model.
    If no silence is found in the search window (rare in real meetings
    but possible in continuous speech), falls back to a hard cut at
    window_sec.
    """
    import numpy as np

    # RMS energy in 50 ms frames.
    frame_size = int(0.05 * SR)
    n_frames = max(1, len(audio) // frame_size)
    trimmed = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt((trimmed.astype(np.float32) ** 2).mean(axis=1) + 1e-12)
    # Silence threshold: 25th percentile of frame RMS (adaptive to
    # the meeting's noise floor — works for quiet rooms and noisy
    # ones alike).
    silence_thresh = float(np.percentile(rms, 25))

    # Search for cut points within the last ~CUT_SEARCH_SEC of each
    # candidate window: prefer an actual low-energy frame inside that
    # search window.
    CUT_SEARCH_SEC = 3.0
    MIN_CHUNK_SEC = 3.0
    search_frames = int(CUT_SEARCH_SEC / 0.05)

    chunks: list[tuple[float, float]] = []
    pos = 0.0
    while pos < duration:
        cur_window = (
            first_window_sec
            if (not chunks and first_window_sec is not None)
            else window_sec
        )
        ideal_end = min(pos + cur_window, duration)
        if ideal_end >= duration:
            chunks.append((pos, ideal_end))
            break
        # Search window: [ideal_end - CUT_SEARCH_SEC, ideal_end] in frames.
        end_frame = int(ideal_end / 0.05)
        start_search = max(int(pos / 0.05) + 1, end_frame - search_frames)
        window_rms = rms[start_search:end_frame]
        if len(window_rms) == 0:
            cut_time = ideal_end
        else:
            min_idx = int(np.argmin(window_rms))
            if float(window_rms[min_idx]) <= silence_thresh:
                cut_frame = start_search + min_idx
                cut_time = cut_frame * 0.05
            else:
                cut_time = ideal_end
            # Refuse to make a chunk shorter than MIN_CHUNK_SEC — avoids
            # pathological cuts when energy is high right at pos.
            if cut_time - pos < MIN_CHUNK_SEC:
                cut_time = ideal_end
        chunks.append((pos, cut_time))
        pos = cut_time
    return chunks


# --- ECAPA helpers (largely unchanged from Tiron_v2) ---
def _non_overlap_window(my_segs, other_segs):
    """Return `(window, is_clean)` — longest crosstalk-free window in
    `my_segs` if any ≥0.5s exists (is_clean=True), else longest raw
    own-segment ≥0.5s (is_clean=False, contaminated by crosstalk).
    Returns (None, False) only when no own segment is ≥0.5s.

    Falling back to a contaminated window matters on overlap-heavy
    meeting audio: many speakers never get a clean 0.5s solo window,
    and excluding them from clustering routes each (ci, lspk) to a
    unique fallback global ID — speaker counts then explode (~135
    fresh IDs on the 75-min neighbours sample). A noisy ECAPA
    embedding still clusters into the right speaker far more often
    than getting a fresh unclustered ID.
    """
    candidates = []
    for seg in my_segs:
        s, e = seg["start"], seg["end"]
        overlaps = sorted(
            (x["start"], x["end"]) for x in other_segs
            if x["end"] > s and x["start"] < e
        )
        if not overlaps:
            candidates.append((s, e))
            continue
        cur = s
        for os_, oe in overlaps:
            if os_ > cur:
                candidates.append((cur, os_))
            cur = max(cur, oe)
        if cur < e:
            candidates.append((cur, e))
    clean = [(s, e) for s, e in candidates if e - s >= 0.5]
    if clean:
        return max(clean, key=lambda p: p[1] - p[0]), True
    raw = [
        (seg["start"], seg["end"])
        for seg in my_segs
        if seg["end"] - seg["start"] >= 0.5
    ]
    if not raw:
        return None, False
    return max(raw, key=lambda p: p[1] - p[0]), False


def embed(arr, start_s, end_s, ecapa):
    import torch
    s = max(0, int(start_s * SR))
    e = min(len(arr), int(end_s * SR))
    if (e - s) / SR < 0.5:
        return None
    clip = torch.from_numpy(arr[s:e]).unsqueeze(0).float().to(_ecapa_device(ecapa))
    with torch.no_grad():
        emb = ecapa.encode_batch(clip).squeeze().cpu().numpy()
    return emb


def _promote_low_mass_to_floor(high, low, mass, min_k_hint):
    """min_speakers hint: promote the largest-mass low clusters to survivors
    (mutating ``high``/``low`` in place) so R2 absorption can't drop K below
    the asserted floor. No-op at the default floor of 1."""
    if min_k_hint > 1:
        while low and len(high) < min_k_hint:
            keep = max(low, key=lambda lab: mass[lab])
            high.add(keep)
            low.discard(keep)


def _spare_weak_to_floor(weak, gid_mass, live_k, min_k_hint):
    """min_speakers hint: demotion deletes clusters, so spare the largest-mass
    weak ones (mutating ``weak`` in place) rather than fall below the asserted
    floor. No-op at the default floor of 1."""
    while weak and live_k - len(weak) < min_k_hint:
        weak.discard(max(weak, key=lambda g: gid_mass.get(g, 0.0)))


def link_speakers_global(
    chunk_segs: list,
    chunk_arrs: list,
    ecapa,
    embed_fn,
    max_k_cap: int | None = None,
    simplified_speaker_linking: bool = False,
    strict_spine_promotion: bool = False,
    low_mass_ecapa_merge: bool = False,
    demote_weak_spine: bool = DEFAULT_DEMOTE_WEAK_SPINE,
    demote_spine_min_mass_sec: float = DEMOTE_SPINE_MIN_MASS_SEC,
    demote_spine_min_windows: int = DEMOTE_SPINE_MIN_WINDOWS,
    cluster_distance_threshold: float | None = None,
    attribution_max_cosine: float | None = None,
    min_k_hint: int = 1,
) -> tuple[dict, dict]:
    """Spine + attribute cross-chunk speaker linking.

    ``min_k_hint`` is the API min_speakers assertion: a hard K floor enforced
    at the dendrogram cut and through R2 mass absorption and weak-spine
    demotion (default 1 = unhinted behavior, exactly).
    The floor applies where clusterable spine evidence exists; the degenerate no-spine and single-chunk paths ignore it rather than fabricate speakers.

    ``cluster_distance_threshold`` / ``attribution_max_cosine`` default to the
    module constants (ECAPA-calibrated). They are overridable so a non-ECAPA
    embedding space (e.g. Sortformer encoder features, whose cosine geometry
    differs) can be calibrated on a dev set without a redeploy.

    Replaces the prior "embed-anything → cluster all → fresh ID for
    anything without an embedding" approach, which exploded speaker
    counts on long meetings (200+ for 1h+ audio: each unembeddable
    (chunk, lspk) got a unique fresh global ID).

    Three stages:
      1. **Spine.** Collect only `(chunk, lspk)` pairs with a clean
         non-overlap window ≥ `MIN_EMBED_SEC`. These are the high-
         confidence evidence. Cluster them into K global speakers
         with one hard count floor and one cleanup pass:
         - R3 (within-chunk hard floor): K >= max distinct DURABLE
           lspks (≥ `MIN_DURABLE_SEC`) in any spine chunk.
         - R2 (mass filter): clusters with total speech below
           `MIN_SPINE_MASS_SEC` are merged into their nearest
           surviving cluster.
         The only ceiling is the explicit API/global speaker cap.
      2. **Centroids.** Compute duration-weighted embedding per surviving cluster.
      3. **Attribute leftovers** (any `(ci, lspk)` not in spine):
         - H1: low-threshold embed (concat all of speaker's own audio
           in chunk). If the nearest centroid passes the cosine
           margin (R5) and absolute distance gate, attribute.
         - H3: most-recent-non-main — pick the spine cluster most
           recently active in prior chunks that isn't a current-
           chunk spine member.
         - H4: merge into the current chunk's main spine speaker.

    Returns `(global_ids, ecapa_windows)` where global_ids maps every
    non-silent (ci, lspk) → integer cluster id (no fresh fallback
    IDs), and ecapa_windows carries per-key diagnostics.
    """
    import numpy as np
    import torch

    cluster_threshold = (
        CLUSTER_DISTANCE_THRESHOLD if cluster_distance_threshold is None
        else float(cluster_distance_threshold)
    )
    attr_max_cosine = (
        ATTRIBUTION_MAX_COSINE if attribution_max_cosine is None
        else float(attribution_max_cosine)
    )

    ecapa_windows: dict = {}
    spine_keys: list = []
    spine_embeddings: list = []

    # --- Stage 1a: collect candidate evidence per (ci, lspk) ---
    for ci, by_spk in enumerate(chunk_segs):
        if not by_spk:
            continue
        for lspk, my_segs in by_spk.items():
            total_dur = sum(s["end"] - s["start"] for s in my_segs)
            others = [s for k, segs in by_spk.items() if k != lspk for s in segs]
            clean_win, is_clean = _non_overlap_window(my_segs, others)
            ecapa_windows[(ci, lspk)] = {
                "total_dur": round(total_dur, 2),
                "in_spine": False,
                "window": None,
                "has_embedding": False,
                "reason": None,
                "attribution": None,
            }
            min_embed_sec = STRICT_SPINE_MIN_EMBED_SEC if strict_spine_promotion else MIN_EMBED_SEC
            min_total_sec = STRICT_SPINE_MIN_TOTAL_SEC if strict_spine_promotion else 0.0
            if (
                is_clean
                and clean_win is not None
                and total_dur >= min_total_sec
                and (clean_win[1] - clean_win[0]) >= min_embed_sec
            ):
                emb = embed_fn(chunk_arrs[ci], *clean_win)
                if emb is not None:
                    spine_keys.append((ci, lspk))
                    spine_embeddings.append(emb)
                    ecapa_windows[(ci, lspk)].update({
                        "in_spine": True,
                        "window": [round(float(clean_win[0]), 2), round(float(clean_win[1]), 2)],
                        "has_embedding": True,
                        "reason": "clean_window",
                    })
                    continue
            if not is_clean:
                ecapa_windows[(ci, lspk)]["reason"] = "no_clean_window"
            elif clean_win is None:
                ecapa_windows[(ci, lspk)]["reason"] = "no_segment_long_enough"
            elif strict_spine_promotion and total_dur < STRICT_SPINE_MIN_TOTAL_SEC:
                ecapa_windows[(ci, lspk)]["reason"] = "strict_total_too_short"
            else:
                ecapa_windows[(ci, lspk)]["reason"] = "clean_window_too_short"

    if strict_spine_promotion and len(spine_keys) < 2:
        print(
            f"[link] strict spine left {len(spine_keys)} clean embeddings; "
            "falling back to normal spine promotion for this request"
        )
        return link_speakers_global(
            chunk_segs,
            chunk_arrs,
            ecapa,
            embed_fn,
            max_k_cap=max_k_cap,
            simplified_speaker_linking=simplified_speaker_linking,
            strict_spine_promotion=False,
            low_mass_ecapa_merge=low_mass_ecapa_merge,
            demote_weak_spine=demote_weak_spine,
            demote_spine_min_mass_sec=demote_spine_min_mass_sec,
            demote_spine_min_windows=demote_spine_min_windows,
            cluster_distance_threshold=cluster_distance_threshold,
            attribution_max_cosine=attribution_max_cosine,
            min_k_hint=min_k_hint,
        )

    # Degenerate case: fewer than 2 spine embeddings — can't cluster.
    # Use model's local_speaker as global id (bounded by requested/global cap).
    # Note this may mis-link across chunks but won't explode K.
    if len(spine_keys) < 2:
        # No (or single) clean ECAPA evidence — we have no signal to
        # link across chunks. Assign each (ci, lspk) a unique gid
        # bounded by the requested/global cap so we don't accidentally
        # merge two different people just because the model happened
        # to pick the same local index in different chunks. This is
        # imperfect but bounded; flag in attribution so downstream
        # consumers can see the meeting was un-linkable. The min_speakers
        # floor deliberately does NOT apply here: with no clusterable
        # evidence, enforcing it would fabricate speaker identities.
        print(
            f"[link] degenerate spine ({len(spine_keys)} clean embeddings across "
            f"{sum(1 for s in chunk_segs if s)} non-silent chunks); "
            f"falling back to per-(chunk, lspk) unique gids"
            + (f"; min_speakers floor ({min_k_hint}) not enforced"
               if min_k_hint > 1 else "")
        )
        global_ids = {}
        fallback_cap = (
            MAX_GLOBAL_SPEAKERS
            if max_k_cap is None
            else max(1, min(int(max_k_cap), MAX_GLOBAL_SPEAKERS))
        )
        next_gid = 0
        for ci, by_spk in enumerate(chunk_segs):
            if not by_spk:
                continue
            for lspk in by_spk.keys():
                global_ids[(ci, lspk)] = next_gid % fallback_cap
                next_gid += 1
                ecapa_windows[(ci, lspk)]["attribution"] = "degenerate_no_spine"
        return global_ids, ecapa_windows

    # --- Stage 1b: cluster the spine ---
    X = np.stack(spine_embeddings)
    n = len(X)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine(X[i], X[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # Per-chunk spine stats for the R3 floor.
    spine_chunks_durable: dict = {}
    spine_durations: list[float] = []  # parallel to spine_keys
    for (ci, lspk) in spine_keys:
        total_dur = sum(s["end"] - s["start"] for s in chunk_segs[ci][lspk])
        spine_durations.append(total_dur)
        if total_dur >= MIN_DURABLE_SEC:
            spine_chunks_durable.setdefault(ci, set()).add(lspk)

    # R3 floor: max distinct durable lspks in any spine chunk. The API
    # min_speakers hint raises the same floor.
    min_k_floor = max((len(s) for s in spine_chunks_durable.values()), default=1)
    min_k_floor = max(min_k_floor, int(min_k_hint))
    oracle_ceiling = (
        MAX_GLOBAL_SPEAKERS if max_k_cap is None else max(1, int(max_k_cap))
    )
    max_k = min(oracle_ceiling, n)
    min_k = min(min_k_floor, max_k)

    # Fixed-threshold agglomerative clustering — cut the dendrogram at
    # CLUSTER_DISTANCE_THRESHOLD cosine. Whatever K falls out is K unless
    # it violates the durable-speaker floor or explicit API/global cap.
    # There is intentionally no per-chunk co-occurrence ceiling: long
    # meetings can have real speakers who never all appear in one chunk.
    spine_labels = _agglomerative_labels(
        dist_mat, distance_threshold=cluster_threshold
    )
    natural_k = len(set(spine_labels))
    if not (min_k <= natural_k <= max_k):
        # Threshold pick was below the durable-speaker floor or above the
        # explicit cap. Re-cluster with the nearest allowed K.
        k_clamped = max(min_k, min(natural_k, max_k))
        spine_labels = _agglomerative_labels(
            dist_mat, n_clusters=k_clamped
        )

    # R2 mass filter: drop phantom clusters with <MIN_SPINE_MASS_SEC speech.
    cluster_mass: dict = {}
    for (ci, lspk), label in zip(spine_keys, spine_labels):
        total_dur = sum(s["end"] - s["start"] for s in chunk_segs[ci][lspk])
        cluster_mass[label] = cluster_mass.get(label, 0.0) + total_dur

    # Compute initial centroids — weighted by total in-chunk speech duration
    # so a sustained local speaker contributes more than a 0.5s scrap. The
    # unweighted mean treats both equally, which lets a small noisy embedding
    # disproportionately pull the centroid.
    durations = np.array(spine_durations, dtype=np.float32)

    def _centroids_from_labels(labels_list):
        idx_by_label: dict = {}
        for i, lab in enumerate(labels_list):
            idx_by_label.setdefault(lab, []).append(i)
        return {
            lab: np.average(X[idxs], axis=0, weights=durations[idxs])
            for lab, idxs in idx_by_label.items()
        }

    centroids = _centroids_from_labels(spine_labels)
    high_mass = {lab for lab, m in cluster_mass.items() if m >= MIN_SPINE_MASS_SEC}
    low_mass = set(centroids.keys()) - high_mass
    _promote_low_mass_to_floor(high_mass, low_mass, cluster_mass, min_k_hint)

    if low_mass and high_mass:
        remap = {}
        for lab in low_mass:
            best_target = min(
                high_mass, key=lambda t: _cosine(centroids[lab], centroids[t])
            )
            remap[lab] = best_target
        spine_labels = [remap.get(l, l) for l in spine_labels]
        centroids = _centroids_from_labels(spine_labels)
    elif low_mass and not high_mass:
        # All clusters are below mass threshold — keep them as-is rather
        # than collapsing to nothing. Means the meeting is genuinely
        # sparse and every cluster matters even if small.
        pass

    # Within-cluster heterogeneity check + post-cluster centroid-merge
    # were prototyped here in PR-bench but reverted: on long
    # (>30 min) meetings, natural within-speaker acoustic drift
    # (calm intro vs heated argument tones) frequently exceeded the
    # SPLIT_HETEROGENEITY_COSINE threshold, causing the dominant
    # speaker to be split into 2+ false clusters (K=5 → K=8 on the
    # 75-min neighbours sample, splitting Pam's 54 min cluster in
    # half). The merge step also rarely fired because the mixed-voice
    # centroid sat too far from any over-split target. These two
    # follow-ups are tracked as future work — they need a smarter
    # heuristic that distinguishes drift from genuine voice change
    # (e.g. silhouette improvement from sub-splitting, or temporal
    # locality of the cluster). For now, the spine + bidirectional
    # H3 design above gives a robust K within ±1 of ground truth on
    # real meetings, and the residual partition errors are documented
    # as a known limitation rather than papered over with
    # over-aggressive splitting.

    global_ids: dict = {
        key: int(label) for key, label in zip(spine_keys, spine_labels)
    }
    for key in spine_keys:
        ecapa_windows[key]["attribution"] = "spine"

    spine_gid_mass: dict[int, float] = {}
    spine_gid_windows: dict[int, int] = {}
    for (ci, lspk), gid in global_ids.items():
        total_dur = sum(s["end"] - s["start"] for s in chunk_segs[ci][lspk])
        gid = int(gid)
        spine_gid_mass[gid] = spine_gid_mass.get(gid, 0.0) + total_dur
        spine_gid_windows[gid] = spine_gid_windows.get(gid, 0) + 1

    if demote_weak_spine and len(set(global_ids.values())) > 1:
        weak_gids = {
            gid for gid, mass in spine_gid_mass.items()
            if mass < demote_spine_min_mass_sec
            or spine_gid_windows.get(gid, 0) < demote_spine_min_windows
        }
        # Keep at least one anchor for H1/H3; otherwise this ablation
        # degenerates into the old per-chunk fallback instead of testing
        # weak-cluster demotion.
        if weak_gids and len(weak_gids) >= len(set(global_ids.values())):
            strongest = max(spine_gid_mass, key=spine_gid_mass.get)
            weak_gids.discard(strongest)
        _spare_weak_to_floor(weak_gids, spine_gid_mass,
                             len(set(global_ids.values())), min_k_hint)
        if weak_gids:
            for key, gid in list(global_ids.items()):
                gid = int(gid)
                if gid in weak_gids:
                    ecapa_windows[key]["demoted_spine"] = {
                        "from_gid": gid,
                        "spine_mass_s": round(spine_gid_mass.get(gid, 0.0), 2),
                        "spine_windows": spine_gid_windows.get(gid, 0),
                        "min_mass_s": demote_spine_min_mass_sec,
                        "min_windows": demote_spine_min_windows,
                    }
                    ecapa_windows[key]["in_spine"] = False
                    ecapa_windows[key]["attribution"] = "demoted_spine_pending"
                    del global_ids[key]
            centroids = {gid: c for gid, c in centroids.items() if int(gid) not in weak_gids}
            spine_gid_mass = {
                gid: mass for gid, mass in spine_gid_mass.items() if gid not in weak_gids
            }

    # --- Stage 3: attribute leftovers ---
    # Precompute per-chunk spine gids (current-chunk spine members are
    # excluded from H3 candidates — the model already says the
    # backchanneler is a different speaker).
    chunk_to_spine_gids: dict = {}
    for (ci, lspk), gid in global_ids.items():
        chunk_to_spine_gids.setdefault(ci, set()).add(gid)

    # Precompute spine occupancy per gid across all chunks. Enables H3
    # bidirectional lookup (prefer-backward, fall-back-forward) so very
    # early backchannels can attribute to a speaker who first appears
    # later in the meeting, instead of merging with the current main
    # (which contradicts the model's "this is a different speaker"
    # signal).
    gid_chunks: dict = {}
    for (ci, lspk), gid in global_ids.items():
        gid_chunks.setdefault(gid, []).append(ci)
    for gid in gid_chunks:
        gid_chunks[gid].sort()

    def _h3_bidirectional(ci_target: int, exclude: set) -> tuple[int | None, str | None]:
        """Pick the spine gid whose nearest occurrence is closest to
        `ci_target` in chunk distance. Backward ties beat forward ties
        (backchannels usually respond to what was just said). Returns
        `(gid, direction_tag)` where direction_tag is the diagnostic
        reason. Returns (None, None) if no candidate exists.
        """
        best_gid: int | None = None
        best_score = float("inf")
        best_tag: str | None = None
        for gid, occ in gid_chunks.items():
            if gid in exclude:
                continue
            # Find nearest occurrence; prefer backward in ties via tiebreaker.
            nearest_back = max((c for c in occ if c <= ci_target), default=None)
            nearest_fwd = min((c for c in occ if c > ci_target), default=None)
            back_dist = (ci_target - nearest_back) if nearest_back is not None else None
            fwd_dist = (nearest_fwd - ci_target) if nearest_fwd is not None else None
            if back_dist is not None and (fwd_dist is None or back_dist <= fwd_dist):
                score = back_dist  # backward strictly preferred on tie
                tag = f"back_chunk={nearest_back}"
            elif fwd_dist is not None:
                score = fwd_dist + 0.1  # nudge forward to lose backward ties
                tag = f"fwd_chunk={nearest_fwd}"
            else:
                continue
            if score < best_score:
                best_score = score
                best_gid = gid
                best_tag = tag
        return best_gid, best_tag

    # Walk chunks in temporal order, attributing each non-spine
    # (ci, lspk) via H1 → H3 → H4. H3 candidate selection reads
    # `gid_chunks` (precomputed across the whole meeting) rather
    # than a streaming "last seen" map, so we can resolve forward
    # references the first time we encounter them.
    for ci, by_spk in enumerate(chunk_segs):
        if not by_spk:
            continue
        for lspk, my_segs in by_spk.items():
            if (ci, lspk) in global_ids:
                continue  # already in spine
            total_dur = sum(s["end"] - s["start"] for s in my_segs)
            attributed: int | None = None

            # H1: low-threshold ECAPA + cosine to spine centroids.
            if total_dur >= 0.3:
                parts = [
                    chunk_arrs[ci][int(s["start"] * SR):int(s["end"] * SR)]
                    for s in my_segs
                ]
                parts = [p for p in parts if len(p) > 0]
                if parts:
                    clip = np.concatenate(parts).astype("float32")
                    if len(clip) / SR >= 0.3:
                        try:
                            clip_t = torch.from_numpy(clip).unsqueeze(0).float().to(
                                _ecapa_device(ecapa)
                            )
                            with torch.no_grad():
                                emb = ecapa.encode_batch(clip_t).squeeze().cpu().numpy()
                            dists = sorted(
                                ((lab, _cosine(emb, c)) for lab, c in centroids.items()),
                                key=lambda x: x[1],
                            )
                            best_lab, best_d = dists[0]
                            second_d = dists[1][1] if len(dists) > 1 else float("inf")
                            margin_ok = (
                                second_d / max(best_d, 1e-6) >= ATTRIBUTION_MARGIN
                                if len(dists) > 1
                                else True
                            )
                            if best_d <= attr_max_cosine and margin_ok:
                                attributed = int(best_lab)
                                ecapa_windows[(ci, lspk)].update({
                                    "has_embedding": True,
                                    "attribution": (
                                        f"H1_d={best_d:.2f}_margin={second_d/max(best_d,1e-6):.2f}"
                                    ),
                                })
                        except RuntimeError as exc:
                            # CUDA OOM / driver faults typically poison
                            # the GPU context for subsequent calls in
                            # this request; surface immediately.
                            raise
                        except Exception as exc:
                            # Non-fatal embedding failure (e.g. malformed
                            # audio chunk). Log and fall through to H3
                            # so we still produce an attribution.
                            print(
                                f"[link] H1 embed failed for c{ci}/l{lspk} "
                                f"({type(exc).__name__}: {exc!r}); falling through to H3"
                            )
                            ecapa_windows[(ci, lspk)]["attribution"] = (
                                f"H1_error_{type(exc).__name__}"
                            )

            # H3: bidirectional nearest-non-main (backward preferred).
            # The simplified path disables this temporal guess. If H1 is
            # not confident, attach the fragment to existing durable spine
            # evidence instead of preserving another possible speaker.
            if attributed is None and not simplified_speaker_linking:
                current_spine = chunk_to_spine_gids.get(ci, set())
                h3_gid, h3_tag = _h3_bidirectional(ci, current_spine)
                if h3_gid is not None:
                    attributed = int(h3_gid)
                    ecapa_windows[(ci, lspk)]["attribution"] = f"H3_{h3_tag}"

            # H4/default fallback. Baseline only reaches this when H3 found
            # no candidate. Simplified linking reaches this for every
            # non-confident leftover, biasing toward fewer extra speakers.
            if attributed is None:
                spine_in_chunk = [
                    (other, sum(s["end"] - s["start"] for s in by_spk[other]))
                    for other in by_spk.keys()
                    if (ci, other) in global_ids
                ]
                if spine_in_chunk:
                    spine_in_chunk.sort(key=lambda x: -x[1])
                    attributed = global_ids[(ci, spine_in_chunk[0][0])]
                    ecapa_windows[(ci, lspk)]["attribution"] = (
                        "SIMPLE_chunk_main" if simplified_speaker_linking else "H4_degenerate_main"
                    )
                elif simplified_speaker_linking and spine_gid_mass:
                    attributed = max(spine_gid_mass, key=spine_gid_mass.get)
                    ecapa_windows[(ci, lspk)]["attribution"] = "SIMPLE_global_main"
                else:
                    attributed = max(global_ids.values(), default=-1) + 1
                    ecapa_windows[(ci, lspk)]["attribution"] = "H4_degenerate_new"

            global_ids[(ci, lspk)] = int(attributed)
            # Deliberately not adding attributed gids to gid_chunks —
            # only spine assignments should drive future H3 lookups,
            # otherwise one bad H1/H3 hop could snowball through the
            # rest of the meeting.

    if low_mass_ecapa_merge:
        # Prototype low-mass cleanup: a low-mass global speaker is merged only
        # when its own spine centroid is close to a durable speaker centroid
        # and clearly better than the second-best durable match. Ambiguous
        # low-mass speakers remain independent.
        gid_mass: dict[int, float] = {}
        for (ci, lspk), gid in global_ids.items():
            total_dur = sum(s["end"] - s["start"] for s in chunk_segs[ci].get(lspk, []))
            gid_mass[int(gid)] = gid_mass.get(int(gid), 0.0) + total_dur
        durable_gids = {gid for gid, mass in gid_mass.items() if mass >= LOW_MASS_MERGE_SEC}
        low_gids = {gid for gid, mass in gid_mass.items() if 0.0 < mass < LOW_MASS_MERGE_SEC}
        merge_gid: dict[int, int] = {}
        low_mass_candidate: dict[int, dict] = {}
        if durable_gids and low_gids:
            for gid in sorted(low_gids):
                if gid not in centroids:
                    low_mass_candidate[gid] = {
                        "from_gid": gid,
                        "from_mass_s": round(gid_mass.get(gid, 0.0), 2),
                        "threshold_s": LOW_MASS_MERGE_SEC,
                        "decision": "keep_no_centroid",
                    }
                    continue
                candidates = [d for d in durable_gids if d in centroids and d != gid]
                if not candidates:
                    low_mass_candidate[gid] = {
                        "from_gid": gid,
                        "from_mass_s": round(gid_mass.get(gid, 0.0), 2),
                        "threshold_s": LOW_MASS_MERGE_SEC,
                        "decision": "keep_no_durable_centroid",
                    }
                    continue
                dists = sorted(
                    ((d, _cosine(centroids[gid], centroids[d])) for d in candidates),
                    key=lambda x: x[1],
                )
                best_gid, best_d = dists[0]
                second_d = dists[1][1] if len(dists) > 1 else float("inf")
                margin = second_d / max(best_d, 1e-6)
                should_merge = best_d <= LOW_MASS_MERGE_MAX_COSINE and margin >= LOW_MASS_MERGE_MARGIN
                low_mass_candidate[gid] = {
                    "from_gid": gid,
                    "to_gid": int(best_gid),
                    "from_mass_s": round(gid_mass.get(gid, 0.0), 2),
                    "threshold_s": LOW_MASS_MERGE_SEC,
                    "best_cosine": round(best_d, 4),
                    "second_cosine": round(second_d, 4) if second_d != float("inf") else None,
                    "margin": round(margin, 4) if margin != float("inf") else None,
                    "max_cosine": LOW_MASS_MERGE_MAX_COSINE,
                    "min_margin": LOW_MASS_MERGE_MARGIN,
                    "decision": "merge" if should_merge else "keep_not_confident",
                }
                if should_merge:
                    merge_gid[gid] = int(best_gid)

        for key, gid in list(global_ids.items()):
            gid = int(gid)
            if gid in low_mass_candidate:
                ecapa_windows[key]["low_mass_candidate"] = low_mass_candidate[gid]

        if merge_gid:
            for key, gid in list(global_ids.items()):
                gid = int(gid)
                if gid in merge_gid:
                    target = merge_gid[gid]
                    global_ids[key] = target
                    ecapa_windows[key]["low_mass_merge"] = {
                        "from_gid": gid,
                        "to_gid": target,
                        "from_mass_s": round(gid_mass.get(gid, 0.0), 2),
                        "threshold_s": LOW_MASS_MERGE_SEC,
                    }
                    old_attr = ecapa_windows[key].get("attribution") or "?"
                    ecapa_windows[key]["attribution"] = f"{old_attr}|LM_to={target}"

    return global_ids, ecapa_windows


def smooth_short_bridge_turns(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Conservatively relabel very short speaker bridges.

    This is a post-hoc experiment, not the default serving behavior. It
    only changes a segment when one short turn is surrounded by the same
    speaker on both sides with tight timing gaps. That tests whether the
    ECAPA-linked global speaker labels can clean obvious local decode
    flips without overriding substantive model turn decisions.
    """
    smoothed = [dict(s) for s in segments]
    changes: list[dict] = []
    for i in range(1, len(smoothed) - 1):
        prev = smoothed[i - 1]
        cur = smoothed[i]
        nxt = smoothed[i + 1]
        if prev.get("speaker") != nxt.get("speaker"):
            continue
        if cur.get("speaker") == prev.get("speaker"):
            continue

        try:
            duration = float(cur["end"]) - float(cur["start"])
            gap_left = float(cur["start"]) - float(prev["end"])
            gap_right = float(nxt["start"]) - float(cur["end"])
        except (KeyError, TypeError, ValueError):
            continue

        words = len((cur.get("text") or "").split())
        if duration > 0.75 or words > 4:
            continue
        if gap_left < -0.05 or gap_right < -0.05:
            continue
        if gap_left > 0.35 or gap_right > 0.35:
            continue

        old_speaker = cur["speaker"]
        cur["speaker"] = prev["speaker"]
        changes.append({
            "index": i,
            "from_speaker": old_speaker,
            "to_speaker": cur["speaker"],
            "start": cur.get("start"),
            "end": cur.get("end"),
            "text": cur.get("text"),
            "reason": "short_bridge_between_same_speaker_context",
        })
    return smoothed, changes


def merge_low_mass_context_turns(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge tiny extra speakers using local temporal context only.

    This is prototype 2 for calibration. It is deliberately separate from
    ECAPA linking: a low-mass speaker must be mostly short turns, and its
    adjacent durable speakers must vote overwhelmingly for one target.
    """
    if not segments:
        return segments, []

    by_gid: dict[int, list[dict]] = {}
    for seg in segments:
        gid = seg.get("_global_id")
        if gid is None:
            continue
        by_gid.setdefault(int(gid), []).append(seg)

    mass_by_gid: dict[int, float] = {}
    for gid, segs in by_gid.items():
        total = 0.0
        for seg in segs:
            try:
                total += max(0.0, float(seg["end"]) - float(seg["start"]))
            except (KeyError, TypeError, ValueError):
                pass
        mass_by_gid[gid] = total

    durable = {gid for gid, mass in mass_by_gid.items() if mass >= LOW_MASS_CONTEXT_SEC}
    low = {gid for gid, mass in mass_by_gid.items() if 0.0 < mass < LOW_MASS_CONTEXT_SEC}
    if not durable or not low:
        return segments, []

    def _is_short_turn(seg: dict) -> bool:
        try:
            dur = max(0.0, float(seg["end"]) - float(seg["start"]))
        except (KeyError, TypeError, ValueError):
            dur = 0.0
        words = len((seg.get("text") or "").split())
        return dur <= LOW_MASS_CONTEXT_MAX_SEG_SEC and words <= LOW_MASS_CONTEXT_MAX_WORDS

    changes: list[dict] = []
    merge_gid: dict[int, int] = {}
    for gid in sorted(low):
        low_segs = by_gid.get(gid, [])
        if not low_segs:
            continue
        short_count = sum(1 for seg in low_segs if _is_short_turn(seg))
        short_turn_ratio = short_count / len(low_segs)
        if short_count != len(low_segs):
            changes.append({
                "from_gid": gid,
                "from_mass_s": round(mass_by_gid.get(gid, 0.0), 2),
                "decision": "keep_long_turns",
                "short_turn_ratio": round(short_turn_ratio, 3),
            })
            continue

        votes: dict[int, int] = {}
        for idx, seg in enumerate(segments):
            if int(seg.get("_global_id", -1)) != gid:
                continue
            neighbor_gids: list[int] = []
            prev = segments[idx - 1] if idx > 0 else None
            nxt = segments[idx + 1] if idx + 1 < len(segments) else None
            if prev is not None:
                pgid = int(prev.get("_global_id", -1))
                gap = float(seg["start"]) - float(prev["end"])
                if pgid in durable and -0.05 <= gap <= LOW_MASS_CONTEXT_MAX_GAP_SEC:
                    neighbor_gids.append(pgid)
            if nxt is not None:
                ngid = int(nxt.get("_global_id", -1))
                gap = float(nxt["start"]) - float(seg["end"])
                if ngid in durable and -0.05 <= gap <= LOW_MASS_CONTEXT_MAX_GAP_SEC:
                    neighbor_gids.append(ngid)
            if len(set(neighbor_gids)) == 1 and neighbor_gids:
                votes[neighbor_gids[0]] = votes.get(neighbor_gids[0], 0) + 1

        total_votes = sum(votes.values())
        if not total_votes:
            changes.append({
                "from_gid": gid,
                "from_mass_s": round(mass_by_gid.get(gid, 0.0), 2),
                "decision": "keep_no_context_votes",
                "short_turn_ratio": round(short_turn_ratio, 3),
            })
            continue
        target, best_votes = max(votes.items(), key=lambda kv: kv[1])
        dominance = best_votes / total_votes
        should_merge = (
            best_votes >= LOW_MASS_CONTEXT_MIN_VOTES
            and dominance >= LOW_MASS_CONTEXT_MIN_DOMINANCE
        )
        changes.append({
            "from_gid": gid,
            "to_gid": int(target),
            "from_mass_s": round(mass_by_gid.get(gid, 0.0), 2),
            "decision": "merge" if should_merge else "keep_weak_context",
            "votes": {str(k): v for k, v in sorted(votes.items())},
            "best_votes": best_votes,
            "total_votes": total_votes,
            "dominance": round(dominance, 3),
            "short_turn_ratio": round(short_turn_ratio, 3),
            "threshold_s": LOW_MASS_CONTEXT_SEC,
        })
        if should_merge:
            merge_gid[gid] = int(target)

    if not merge_gid:
        return segments, changes

    merged = [dict(seg) for seg in segments]
    for seg in merged:
        gid = int(seg.get("_global_id", -1))
        if gid in merge_gid:
            target = merge_gid[gid]
            seg["_global_id"] = target
            seg["speaker"] = f"SPEAKER_{target:02d}"
            seg["_low_mass_context_merge"] = {"from_gid": gid, "to_gid": target}
    return merged, changes


# ============================================================================
# Two-pass calibrated speaker linking.
#
# These functions operate on "parsed pass" dictionaries:
#   {"chunk_windows": [(s,e),...],
#    "chunk_segs":   [ {lspk:int -> [ {start,end,text,...}, ...]}, ... ],
#    "emb":          { (ci,lspk) -> {total_dur, is_clean, spine_emb|None,
#                                    concat_emb|None} }}
# Pass A is the canonical transcript; pass B (staggered grid) contributes
# evidence only.
# ============================================================================

CAL_WITNESS_MIN_CO_SEC = 0.5    # min co-attributed seconds for a witness match
CAL_WITNESS_EMB_GUARD = 0.65    # reject witness match above this cosine
CAL_WITNESS_MARGIN_SEC = 0.25   # winner must beat runner-up co-time by this
CAL_CLIP_LO, CAL_CLIP_HI = 0.35, 0.95
ECAPA_EMBED_BATCH_SIZE = 16


def _encode_ecapa_clips_batched(ecapa, clips, batch_size: int = ECAPA_EMBED_BATCH_SIZE):
    """Encode variable-length ECAPA clips in device-local micro-batches.

    ``clips`` is a list of 1-D float32 numpy arrays. The returned list is
    parallel to ``clips``. Padding is masked with SpeechBrain's ``wav_lens`` so
    batched embeddings match the single-clip path as closely as possible.
    """
    import numpy as np
    import torch

    out = [None] * len(clips)
    jobs = [
        (idx, np.asarray(clip, dtype=np.float32))
        for idx, clip in enumerate(clips)
        if clip is not None and len(clip) > 0
    ]
    if not jobs:
        return out

    def _encode_range(batch_jobs, current_batch_size: int):
        for start in range(0, len(batch_jobs), current_batch_size):
            batch = batch_jobs[start:start + current_batch_size]
            try:
                max_len = max(len(clip) for _, clip in batch)
                wavs = torch.zeros((len(batch), max_len), dtype=torch.float32, device=_ecapa_device(ecapa))
                wav_lens = torch.empty((len(batch),), dtype=torch.float32, device=_ecapa_device(ecapa))
                for row, (_, clip) in enumerate(batch):
                    wavs[row, :len(clip)] = torch.from_numpy(clip).to(device=_ecapa_device(ecapa))
                    wav_lens[row] = len(clip) / max_len
                with torch.no_grad():
                    encoded = ecapa.encode_batch(wavs, wav_lens=wav_lens)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(batch) == 1:
                    raise
                try:
                    del wavs, wav_lens
                except UnboundLocalError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                smaller = max(1, len(batch) // 2)
                print(
                    f"[tiron] ECAPA batch OOM at {len(batch)} clips; "
                    f"retrying with batch_size={smaller}"
                )
                _encode_range(batch, smaller)
                continue
            arr = encoded.detach().cpu().numpy()
            arr = np.asarray(arr).reshape(len(batch), -1)
            for (idx, _), emb in zip(batch, arr):
                out[idx] = emb
            del wavs, wav_lens, encoded

    _encode_range(jobs, max(1, int(batch_size)))
    return out


def compute_node_embeddings(chunk_segs, chunk_arrs, ecapa, embed_fn) -> dict:
    """Per-(ci, lspk) embeddings, mirroring link_speakers_global's evidence:
    the stage-1a clean-window spine embedding and the H1 concat embedding.
    ECAPA clips are encoded in micro-batches for long two-pass meetings.
    """
    import numpy as np

    emb: dict = {}
    spine_jobs: list[tuple[tuple[int, int], np.ndarray]] = []
    concat_jobs: list[tuple[tuple[int, int], np.ndarray]] = []
    # Preserve the old embed_fn contract for custom embedding backends. The
    # batched path is only for the standard ECAPA helpers used by serving.
    can_batch_spine = getattr(embed_fn, "__name__", "") in {"embed", "_embed"}

    for ci, by_spk in enumerate(chunk_segs):
        for lspk, my_segs in (by_spk or {}).items():
            key = (ci, lspk)
            total_dur = sum(s["end"] - s["start"] for s in my_segs)
            others = [s for k, segs in by_spk.items() if k != lspk for s in segs]
            clean_win, is_clean = _non_overlap_window(my_segs, others)
            emb[key] = {
                "total_dur": total_dur,
                "is_clean": bool(is_clean),
                "spine_emb": None,
                "concat_emb": None,
            }
            if is_clean and clean_win is not None:
                if can_batch_spine:
                    s = max(0, int(clean_win[0] * SR))
                    e = min(len(chunk_arrs[ci]), int(clean_win[1] * SR))
                    if (e - s) / SR >= 0.5:
                        spine_jobs.append((key, chunk_arrs[ci][s:e].astype("float32")))
                else:
                    e = embed_fn(chunk_arrs[ci], *clean_win)
                    if e is not None:
                        emb[key]["spine_emb"] = e
            if total_dur >= 0.3:
                parts = [
                    chunk_arrs[ci][int(s["start"] * SR):int(s["end"] * SR)]
                    for s in my_segs
                ]
                parts = [p for p in parts if len(p) > 0]
                if parts:
                    clip = np.concatenate(parts).astype("float32")
                    if len(clip) / SR >= 0.3:
                        concat_jobs.append((key, clip))

    if spine_jobs:
        spine_embs = _encode_ecapa_clips_batched(ecapa, [clip for _, clip in spine_jobs])
        for (key, _), e in zip(spine_jobs, spine_embs):
            if e is not None:
                emb[key]["spine_emb"] = e

    if concat_jobs:
        try:
            concat_embs = _encode_ecapa_clips_batched(ecapa, [clip for _, clip in concat_jobs])
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[tiron] concat embed batch failed: {exc!r}; retrying serially")
            concat_embs = []
            for _, clip in concat_jobs:
                try:
                    concat_embs.extend(_encode_ecapa_clips_batched(ecapa, [clip], batch_size=1))
                except RuntimeError:
                    raise
                except Exception as single_exc:  # noqa: BLE001
                    print(f"[tiron] concat embed failed: {single_exc!r}")
                    concat_embs.append(None)
        for (key, _), e in zip(concat_jobs, concat_embs):
            if e is not None:
                emb[key]["concat_emb"] = e

    return emb


def _cal_abs_ivals(parsed, ci, lspk):
    cs, _ = parsed["chunk_windows"][ci]
    return [(cs + s["start"], cs + s["end"]) for s in parsed["chunk_segs"][ci][lspk]]


def _cal_co_time(ivals_a, ivals_b):
    total = 0.0
    for s1, e1 in ivals_a:
        for s2, e2 in ivals_b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def _cal_witness_pairs(parsed_a, parsed_b):
    """Cross-pass same-speaker matches: per overlapping (A,B) window pair,
    Hungarian on co-attributed seconds, gated by absolute co-time, a
    runner-up margin, and an embedding sanity guard."""
    import numpy as np

    pairs = set()
    ivals_a = {
        (ci, l): _cal_abs_ivals(parsed_a, ci, l)
        for ci, by in enumerate(parsed_a["chunk_segs"]) for l in (by or {})
    }
    ivals_b = {
        (ci, l): _cal_abs_ivals(parsed_b, ci, l)
        for ci, by in enumerate(parsed_b["chunk_segs"]) for l in (by or {})
    }
    for ci_a, (sa, ea) in enumerate(parsed_a["chunk_windows"]):
        spk_a = sorted((parsed_a["chunk_segs"][ci_a] or {}).keys())
        if not spk_a:
            continue
        for ci_b, (sb, eb) in enumerate(parsed_b["chunk_windows"]):
            if eb <= sa or sb >= ea:
                continue
            spk_b = sorted((parsed_b["chunk_segs"][ci_b] or {}).keys())
            if not spk_b:
                continue
            co = np.zeros((len(spk_a), len(spk_b)))
            for i, la in enumerate(spk_a):
                for j, lb in enumerate(spk_b):
                    co[i, j] = _cal_co_time(ivals_a[(ci_a, la)], ivals_b[(ci_b, lb)])
            rows, cols = _linear_sum_assignment(-co)
            for i, j in zip(rows, cols):
                best = co[i, j]
                if best <= 0.0 or best < CAL_WITNESS_MIN_CO_SEC:
                    continue
                runner_row = max((co[i, jj] for jj in range(len(spk_b)) if jj != j), default=0.0)
                runner_col = max((co[ii, j] for ii in range(len(spk_a)) if ii != i), default=0.0)
                if best - max(runner_row, runner_col) < CAL_WITNESS_MARGIN_SEC:
                    continue
                ea_emb = parsed_a["emb"].get((ci_a, spk_a[i])) or {}
                eb_emb = parsed_b["emb"].get((ci_b, spk_b[j])) or {}
                ca, cb = ea_emb.get("concat_emb"), eb_emb.get("concat_emb")
                if ca is not None and cb is not None and _cosine(ca, cb) > CAL_WITNESS_EMB_GUARD:
                    continue
                pairs.add(((ci_a, spk_a[i]), (ci_b, spk_b[j])))
    return pairs


def calibrate_cluster_threshold(parsed_a, parsed_b, pairs, min_samples):
    """Per-meeting threshold: q75 of same-speaker distances (witness pairs,
    both sides durable + spine-grade) and q25 of cross-speaker distances
    (distinct durable locals within one window, both passes); cut at the
    midpoint, clipped. Returns (t|None, n_same, n_cross)."""
    import numpy as np

    same = []
    for (ka, kb) in pairs:
        ea = parsed_a["emb"].get(ka) or {}
        eb = parsed_b["emb"].get(kb) or {}
        if ea.get("spine_emb") is None or eb.get("spine_emb") is None:
            continue
        if (ea.get("total_dur", 0.0) < MIN_DURABLE_SEC
                or eb.get("total_dur", 0.0) < MIN_DURABLE_SEC):
            continue
        same.append(_cosine(ea["spine_emb"], eb["spine_emb"]))
    cross = []
    for parsed in (parsed_a, parsed_b):
        for ci, by in enumerate(parsed["chunk_segs"]):
            durable = [
                l for l in (by or {})
                if (parsed["emb"].get((ci, l)) or {}).get("total_dur", 0.0) >= MIN_DURABLE_SEC
                and (parsed["emb"].get((ci, l)) or {}).get("spine_emb") is not None
            ]
            for x in range(len(durable)):
                for y in range(x + 1, len(durable)):
                    cross.append(_cosine(
                        parsed["emb"][(ci, durable[x])]["spine_emb"],
                        parsed["emb"][(ci, durable[y])]["spine_emb"]))
    if len(same) < min_samples or len(cross) < min_samples:
        return None, len(same), len(cross)
    t = (float(np.quantile(same, 0.75)) + float(np.quantile(cross, 0.25))) / 2.0
    return float(np.clip(t, CAL_CLIP_LO, CAL_CLIP_HI)), len(same), len(cross)


def two_pass_degraded_diag(exc: BaseException) -> dict:
    """Diagnostic payload when two-pass calibrated linking fails and the request
    silently degrades to the single-pass linker.

    The degradation is invisible to callers unless they inspect
    ``result["two_pass"]``; ``degraded=True`` makes the fallback explicit and
    ``oom=True`` flags CUDA out-of-memory specifically so prod log alerts can
    distinguish memory-pressure degradation from logic errors.
    """
    name = type(exc).__name__
    is_oom = "OutOfMemory" in name or "out of memory" in str(exc).lower()
    return {
        "engaged": False,
        "degraded": True,
        "oom": is_oom,
        "error": f"{name}: {exc}"[:200],
    }


def link_speakers_calibrated(
    parsed_a: dict,
    parsed_b: dict,
    max_k_cap: int | None = None,
    min_calib_samples: int = 30,
    demote_mass_sec: float = 10.0,
    demote_windows: int = 2,
    min_k_hint: int = 1,
):
    """Two-pass calibrated cross-window linking (v10 config).

    Returns ``(global_ids, ecapa_windows, diag)`` covering every non-silent
    pass-A (ci, lspk), or ``(None, None, diag)`` when the meeting yields too
    few calibration samples — caller falls back to the single-pass linker.

    ``min_k_hint`` is the API min_speakers assertion: a hard K floor enforced
    at the dendrogram cut AND through R2 mass absorption, weak-spine demotion,
    and the must-link merge loop (heuristic floors only shape the cut). The
    default of 1 reproduces unhinted behavior exactly.
    The floor applies where clusterable spine evidence exists; the degenerate no-spine and single-chunk paths ignore it rather than fabricate speakers.
    """
    import numpy as np

    chunk_segs = parsed_a["chunk_segs"]
    emb = parsed_a["emb"]
    diag: dict = {"engaged": False}

    pairs = _cal_witness_pairs(parsed_a, parsed_b)
    t, n_same, n_cross = calibrate_cluster_threshold(
        parsed_a, parsed_b, pairs, min_calib_samples)
    diag.update({"n_same": n_same, "n_cross": n_cross,
                 "threshold": round(t, 4) if t is not None else None})
    if t is None:
        return None, None, diag
    diag["engaged"] = True

    # --- spine collection (stage 1a of link_speakers_global) ---
    spine_keys, spine_X, spine_dur = [], [], []
    for ci, by_spk in enumerate(chunk_segs):
        for lspk in (by_spk or {}):
            e = emb.get((ci, lspk)) or {}
            if e.get("is_clean") and e.get("spine_emb") is not None:
                spine_keys.append((ci, lspk))
                spine_X.append(e["spine_emb"])
                spine_dur.append(e["total_dur"])
    if len(spine_keys) < 2:
        # Degenerate spine — caller falls back to the single-pass linker, so
        # the diag must not report the calibrated path as engaged.
        diag["engaged"] = False
        return None, None, diag

    X = np.stack(spine_X)
    durations = np.asarray(spine_dur, dtype=np.float32)
    n = len(X)

    # K floor: pass-A durable distinctness (R3) + pass-B durable distinctness
    # (a 30s B window straddles two A windows and can see across boundaries).
    durable_by_chunk: dict = {}
    for (ci, lspk), d in zip(spine_keys, spine_dur):
        if d >= MIN_DURABLE_SEC:
            durable_by_chunk.setdefault(ci, set()).add(lspk)
    min_k = max((len(s) for s in durable_by_chunk.values()), default=1)
    b_floor = 1
    for ci, by in enumerate(parsed_b["chunk_segs"]):
        durable = [
            l for l in (by or {})
            if (parsed_b["emb"].get((ci, l)) or {}).get("total_dur", 0.0) >= MIN_DURABLE_SEC
        ]
        b_floor = max(b_floor, len(durable))
    min_k = max(min_k, b_floor, int(min_k_hint))
    max_k = min(MAX_GLOBAL_SPEAKERS if max_k_cap is None else max(1, int(max_k_cap)), n)
    min_k = min(min_k, max_k)

    # Cannot-links: A-spine nodes witness-matched to DISTINCT durable locals
    # of one B window are different speakers. Provenance (which B windows)
    # retained for demotion protection.
    a_partner: dict = {}
    for (ka, kb) in pairs:
        a_partner.setdefault(kb, []).append(ka)
    cannot: dict = {}
    for ci_b, by in enumerate(parsed_b["chunk_segs"]):
        durable = sorted(
            l for l in (by or {})
            if (parsed_b["emb"].get((ci_b, l)) or {}).get("total_dur", 0.0) >= MIN_DURABLE_SEC
        )
        for x in range(len(durable)):
            for y in range(x + 1, len(durable)):
                for ka1 in a_partner.get((ci_b, durable[x]), []):
                    for ka2 in a_partner.get((ci_b, durable[y]), []):
                        if ka1 != ka2:
                            cannot.setdefault((ka1, ka2), set()).add(ci_b)

    # --- clustering at the calibrated threshold; trust the natural cut ---
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine(X[i], X[j])
            dist[i, j] = dist[j, i] = d
    key_idx = {k: i for i, k in enumerate(spine_keys)}
    cpairs = []
    for ka, kb in cannot:
        ia, ib = key_idx.get(ka), key_idx.get(kb)
        if ia is None or ib is None or ia == ib:
            continue
        cpairs.append((ia, ib))
        dist[ia, ib] = dist[ib, ia] = max(dist[ia, ib], 1.0)

    labels = _agglomerative_labels(dist, distance_threshold=t)
    natural_k = len(set(labels))
    if natural_k < min_k:
        labels = _agglomerative_labels(dist, n_clusters=min_k)
    # Split clusters violating a cannot-link by re-clustering at higher K.
    k = len(set(labels))
    while k < max_k and any(labels[i] == labels[j] for i, j in cpairs):
        k += 1
        labels = _agglomerative_labels(dist, n_clusters=k)

    # R2 mass merge (phantoms < MIN_SPINE_MASS_SEC into nearest survivor).
    def _centroids(labels_list):
        out = {}
        by_lab: dict = {}
        for i, lab in enumerate(labels_list):
            by_lab.setdefault(lab, []).append(i)
        for lab, idxs in by_lab.items():
            out[lab] = np.average(X[idxs], axis=0, weights=durations[idxs])
        return out

    mass: dict = {}
    for lab, d in zip(labels, spine_dur):
        mass[lab] = mass.get(lab, 0.0) + d
    high = {lab for lab, m in mass.items() if m >= MIN_SPINE_MASS_SEC}
    low = set(mass) - high
    _promote_low_mass_to_floor(high, low, mass, min_k_hint)
    if low and high:
        cents = _centroids(labels)
        remap = {
            lab: min(high, key=lambda tgt: _cosine(cents[lab], cents[tgt]))
            for lab in low
        }
        labels = [remap.get(l, l) for l in labels]
    centroids = _centroids(labels)
    global_ids = {k_: int(l) for k_, l in zip(spine_keys, labels)}

    # Weak-spine demotion (calibrated-path thresholds), with cannot-evidence
    # protection: a weak cluster asserted distinct by >=2 independent B
    # windows (and >=5s mass) survives.
    gid_mass: dict = {}
    gid_windows: dict = {}
    for (ci, lspk), gid in global_ids.items():
        gid_mass[gid] = gid_mass.get(gid, 0.0) + emb[(ci, lspk)]["total_dur"]
        gid_windows[gid] = gid_windows.get(gid, 0) + 1
    if len(set(global_ids.values())) > 1:
        weak = {g for g, m in gid_mass.items()
                if m < demote_mass_sec or gid_windows.get(g, 0) < demote_windows}
        if weak and cannot:
            wins_by_gid: dict = {}
            for (ka, kb), prov in cannot.items():
                ga, gb = global_ids.get(ka), global_ids.get(kb)
                if ga is None or gb is None or ga == gb:
                    continue
                if ga in weak and gb not in weak:
                    wins_by_gid.setdefault(ga, set()).update(prov)
                if gb in weak and ga not in weak:
                    wins_by_gid.setdefault(gb, set()).update(prov)
            protected = {
                g for g, wins in wins_by_gid.items()
                if len(wins) >= 2 and gid_mass.get(g, 0.0) >= 5.0
            }
            weak -= protected
        if weak and len(weak) >= len(set(global_ids.values())):
            weak.discard(max(gid_mass, key=gid_mass.get))
        _spare_weak_to_floor(weak, gid_mass,
                             len(set(global_ids.values())), min_k_hint)
        if weak:
            global_ids = {k_: g for k_, g in global_ids.items() if g not in weak}
            centroids = {g: c for g, c in centroids.items() if g not in weak}

    # Global cap AFTER demotion: merge smallest-mass survivor into nearest.
    cap = MAX_GLOBAL_SPEAKERS if max_k_cap is None else max(1, int(max_k_cap))
    while len(set(global_ids.values())) > cap:
        live_mass: dict = {}
        for (ci, lspk), g in global_ids.items():
            live_mass[g] = live_mass.get(g, 0.0) + emb[(ci, lspk)]["total_dur"]
        smallest = min(live_mass, key=live_mass.get)
        others = [g for g in live_mass if g != smallest and g in centroids]
        if not others or smallest not in centroids:
            break
        target = min(others, key=lambda g: _cosine(centroids[smallest], centroids[g]))
        global_ids = {k_: (target if g == smallest else g) for k_, g in global_ids.items()}
        centroids.pop(smallest, None)

    # Must-link merges: any surviving pair under the calibrated threshold,
    # vote-backed pairs first, cannot-veto throughout.
    co_w = {(ka, kb): _cal_co_time(_cal_abs_ivals(parsed_a, *ka), _cal_abs_ivals(parsed_b, *kb))
            for (ka, kb) in pairs}
    cannot_gid = {
        frozenset((global_ids[a], global_ids[b]))
        for (a, b) in cannot if a in global_ids and b in global_ids
        and global_ids[a] != global_ids[b]
    }
    changed = True
    while changed:
        changed = False
        # min_speakers hint: each must-link merge drops live K by one — stop
        # at the asserted floor. (No-op at the default floor of 1: a single
        # live cluster has no pairs to merge.)
        live_ids = set(global_ids.values())
        if len(live_ids) <= min_k_hint:
            break
        votes_by_pair: dict = {}
        b_partner_g: dict = {}
        for (ka, kb) in pairs:
            gid = global_ids.get(ka)
            if gid is not None:
                b_partner_g.setdefault(kb, {})
                b_partner_g[kb][gid] = b_partner_g[kb].get(gid, 0.0) + co_w[(ka, kb)]
        for kb, by_gid in b_partner_g.items():
            gs = sorted(by_gid)
            for x in range(len(gs)):
                for y in range(x + 1, len(gs)):
                    pk = frozenset((gs[x], gs[y]))
                    w = min(by_gid[gs[x]], by_gid[gs[y]])
                    votes_by_pair[pk] = votes_by_pair.get(pk, 0.0) + w
        live = sorted(live_ids)
        all_pairs = {frozenset((g1, g2)) for i, g1 in enumerate(live) for g2 in live[i + 1:]}
        candidates = sorted(((p, votes_by_pair.get(p, 0.0)) for p in all_pairs),
                            key=lambda kv: -kv[1])
        for pair_key, _w in candidates:
            if pair_key in cannot_gid:
                continue
            g1, g2 = sorted(pair_key)
            if g1 not in centroids or g2 not in centroids:
                continue
            if _cosine(centroids[g1], centroids[g2]) >= t:
                continue
            global_ids = {k_: (g1 if g == g2 else g) for k_, g in global_ids.items()}
            members = [k_ for k_, g in global_ids.items() if g == g1
                       and emb[k_].get("spine_emb") is not None]
            if members:
                embs = np.stack([emb[k_]["spine_emb"] for k_ in members])
                ws = np.asarray([max(emb[k_]["total_dur"], 1e-3) for k_ in members])
                centroids[g1] = np.average(embs, axis=0, weights=ws)
            centroids.pop(g2, None)
            cannot_gid = {frozenset((g1 if g == g2 else g) for g in p) for p in cannot_gid}
            cannot_gid = {p for p in cannot_gid if len(p) == 2}
            changed = True
            break

    # Chain votes for leftovers (spine gid -> B node -> leftover A node).
    b_votes: dict = {}
    for (ka, kb) in pairs:
        gid = global_ids.get(ka)
        if gid is not None:
            b_votes.setdefault(kb, {})
            b_votes[kb][gid] = b_votes[kb].get(gid, 0.0) + co_w[(ka, kb)]
    chain_votes: dict = {}
    for (ka, kb) in pairs:
        if ka in global_ids:
            continue
        for gid, w in (b_votes.get(kb) or {}).items():
            chain_votes.setdefault(ka, {})
            chain_votes[ka][gid] = chain_votes[ka].get(gid, 0.0) + min(w, co_w[(ka, kb)])

    # Leftover attribution: H1 (concat emb, widened gate) -> chain votes ->
    # H3 bidirectional temporal -> H4 chunk-main.
    attr_max = max(ATTRIBUTION_MAX_COSINE, t)
    ecapa_windows: dict = {}
    for key in global_ids:
        ecapa_windows[key] = {"in_spine": True, "attribution": "cal_spine"}
    chunk_to_spine_gids: dict = {}
    gid_chunks: dict = {}
    for (ci, lspk), gid in global_ids.items():
        chunk_to_spine_gids.setdefault(ci, set()).add(gid)
        gid_chunks.setdefault(gid, []).append(ci)
    for g in gid_chunks:
        gid_chunks[g].sort()

    def _h3(ci_target, exclude):
        best_gid, best_score = None, float("inf")
        for gid, occ in gid_chunks.items():
            if gid in exclude:
                continue
            nb = max((c for c in occ if c <= ci_target), default=None)
            nf = min((c for c in occ if c > ci_target), default=None)
            bd = (ci_target - nb) if nb is not None else None
            fd = (nf - ci_target) if nf is not None else None
            if bd is not None and (fd is None or bd <= fd):
                score = bd
            elif fd is not None:
                score = fd + 0.1
            else:
                continue
            if score < best_score:
                best_score, best_gid = score, gid
        return best_gid

    for ci, by_spk in enumerate(chunk_segs):
        for lspk in (by_spk or {}):
            if (ci, lspk) in global_ids:
                continue
            e = emb.get((ci, lspk)) or {}
            attributed = None
            attribution = None
            cemb = e.get("concat_emb")
            if cemb is not None and centroids:
                dists = sorted(
                    ((lab, _cosine(cemb, c)) for lab, c in centroids.items()),
                    key=lambda x: x[1],
                )
                best_lab, best_d = dists[0]
                second_d = dists[1][1] if len(dists) > 1 else float("inf")
                margin_ok = (second_d / max(best_d, 1e-6) >= ATTRIBUTION_MARGIN) if len(dists) > 1 else True
                if best_d <= attr_max and margin_ok:
                    attributed = int(best_lab)
                    attribution = f"cal_H1_d={best_d:.2f}"
            if attributed is None and chain_votes.get((ci, lspk)):
                votes = chain_votes[(ci, lspk)]
                attributed = int(max(votes.items(), key=lambda kv: kv[1])[0])
                attribution = "cal_chain"
            if attributed is None:
                h3 = _h3(ci, chunk_to_spine_gids.get(ci, set()))
                if h3 is not None:
                    attributed = int(h3)
                    attribution = "cal_H3"
            if attributed is None:
                spine_here = [
                    (o, emb[(ci, o)]["total_dur"]) for o in by_spk
                    if (ci, o) in global_ids
                ]
                if spine_here:
                    attributed = global_ids[(ci, max(spine_here, key=lambda x: x[1])[0])]
                    attribution = "cal_H4_main"
                else:
                    attributed = max(global_ids.values(), default=-1) + 1
                    attribution = "cal_H4_new"
            global_ids[(ci, lspk)] = int(attributed)
            ecapa_windows[(ci, lspk)] = {"in_spine": False, "attribution": attribution}

    diag["final_k"] = len(set(global_ids.values()))
    return global_ids, ecapa_windows, diag


def link_with_optional_two_pass(
    *,
    arr,
    duration: float,
    chunks,
    chunk_segs,
    chunk_arrs,
    decode_and_parse,
    ecapa,
    embed_fn,
    use_two_pass: bool,
    window_sec: float = 30.0,
    pass_b_window_sec: float | None = None,
    pass_b_first_window_sec: float | None = None,
    min_calib_samples: int = 30,
    cal_demote_mass_sec: float = 10.0,
    cal_demote_windows: int = 2,
    max_k_cap: int | None = None,
    simplified_speaker_linking: bool = False,
    strict_spine_promotion: bool = False,
    low_mass_ecapa_merge: bool = False,
    demote_weak_spine: bool = True,
    demote_spine_min_mass_sec: float = 10.0,
    demote_spine_min_windows: int = 2,
    pad_chunks_to_samples: int | None = None,
    timings: dict | None = None,
    log_prefix: str = "[pipeline]",
    min_k_hint: int = 1,
):
    """Cross-window speaker linking with two-pass calibration as the DEFAULT path.

    The optional second pass decodes a staggered grid via the supplied
    ``decode_and_parse(chunk_arrs, chunk_durations)``
    callback, calibrate the clustering threshold from cross-grid witness pairs
    (link_speakers_calibrated), and fall back to the single-pass linker only
    when the meeting is under-evidenced or two-pass is disabled — the same
    degradation contract as the single-pass path.

    ``window_sec`` is the model's absolute decode window (30s for current
    models). Pass-B grid defaults scale with it (5/6 window, half-window first
    chunk) and reproduce serving's 25s/15s exactly at 30s; explicit
    Explicit ``pass_b_*`` values override the scaled defaults.

    Returns ``(global_ids, ecapa_windows, diag)``; ``diag`` carries the
    calibration diagnostics (``engaged`` False on fallback).
    """
    import time

    import numpy as np

    timings = timings if timings is not None else {}
    diag: dict = {"engaged": False, "mode": "single_pass"}

    pass_b = None
    if use_two_pass and len(chunks) >= 5 \
            and sum(1 for by_spk in chunk_segs if by_spk) > 1:
        t_b = time.time()
        b_win = pass_b_window_sec if pass_b_window_sec is not None else window_sec * (25.0 / 30.0)
        b_first = pass_b_first_window_sec if pass_b_first_window_sec is not None else window_sec / 2.0
        chunks_b = fixed_window_chunks(arr, duration, window_sec=b_win, first_window_sec=b_first)
        chunk_arrs_b = [arr[int(cs * SR):int(ce * SR)] for cs, ce in chunks_b]
        chunk_durations_b = [float(ce - cs) for cs, ce in chunks_b]
        if pad_chunks_to_samples:
            for i, ca in enumerate(chunk_arrs_b):
                if len(ca) < pad_chunks_to_samples:
                    chunk_arrs_b[i] = np.concatenate(
                        [ca, np.zeros(pad_chunks_to_samples - len(ca), dtype=ca.dtype)])
        chunk_segs_b, _ = decode_and_parse(chunk_arrs_b, chunk_durations_b)
        pass_b = {"chunks": chunks_b, "chunk_segs": chunk_segs_b, "chunk_arrs": chunk_arrs_b}
        timings["pass_b_decode_s"] = round(time.time() - t_b, 3)

    if pass_b is not None:
        t_cal = time.time()
        try:
            parsed_a = {
                "chunk_windows": [tuple(c) for c in chunks],
                "chunk_segs": chunk_segs,
                "emb": compute_node_embeddings(chunk_segs, chunk_arrs, ecapa, embed_fn),
            }
            parsed_b = {
                "chunk_windows": [tuple(c) for c in pass_b["chunks"]],
                "chunk_segs": pass_b["chunk_segs"],
                "emb": compute_node_embeddings(pass_b["chunk_segs"], pass_b["chunk_arrs"], ecapa, embed_fn),
            }
            cal_ids, cal_windows, cal_diag = link_speakers_calibrated(
                parsed_a, parsed_b,
                max_k_cap=max_k_cap,
                min_calib_samples=min_calib_samples,
                demote_mass_sec=cal_demote_mass_sec,
                demote_windows=cal_demote_windows,
                min_k_hint=min_k_hint,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, don't fail the request
            cal_ids, cal_windows = None, None
            cal_diag = two_pass_degraded_diag(exc)
            # The alert makes a transparent fallback observable to callers.
            print(f"{log_prefix} ALERT two-pass linking degraded to single-pass "
                  f"(oom={cal_diag['oom']}): {type(exc).__name__}: {exc!r}", flush=True)
        pass_b["chunk_arrs"] = None
        if "parsed_a" in locals():
            del parsed_a
        if "parsed_b" in locals():
            del parsed_b
        timings["calibrated_link_s"] = round(time.time() - t_cal, 3)
        print(f"{log_prefix} two-pass calibrated: {cal_diag}", flush=True)
        if cal_ids is not None:
            diag = dict(cal_diag or {})
            diag["mode"] = "two_pass_calibrated"
            return cal_ids, cal_windows or {}, diag
        diag.update(cal_diag or {})
        diag["mode"] = "single_pass_fallback"

    t0 = time.time()
    if sum(1 for by_spk in chunk_segs if by_spk) <= 1:
        # Single non-silent chunk: no cross-chunk linking exists, so the
        # speaker-count hints deliberately do NOT apply here — the model's
        # within-chunk labels are emitted as-is (forcing a min floor would
        # fabricate speakers; max already caps the decode grammar upstream).
        if min_k_hint > 1:
            print(f"{log_prefix} single-chunk request: min_speakers floor "
                  f"({min_k_hint}) not applicable to within-chunk labels")
        global_ids: dict = {}
        ecapa_windows: dict = {}
        for ci, by_spk in enumerate(chunk_segs):
            if not by_spk:
                continue
            for lspk in by_spk.keys():
                global_ids[(ci, lspk)] = int(lspk) - 1
                ecapa_windows[(ci, lspk)] = {
                    "in_spine": False, "has_embedding": False,
                    "attribution": "single_chunk_local",
                }
        timings["ecapa_s"] = 0.0
        timings["cluster_s"] = round(time.time() - t0, 3)
        return global_ids, ecapa_windows, diag

    global_ids, ecapa_windows = link_speakers_global(
        chunk_segs, chunk_arrs, ecapa, embed_fn,
        max_k_cap=max_k_cap,
        simplified_speaker_linking=simplified_speaker_linking,
        strict_spine_promotion=strict_spine_promotion,
        low_mass_ecapa_merge=low_mass_ecapa_merge,
        demote_weak_spine=demote_weak_spine,
        demote_spine_min_mass_sec=demote_spine_min_mass_sec,
        demote_spine_min_windows=demote_spine_min_windows,
        min_k_hint=min_k_hint,
    )
    timings["ecapa_s"] = round(time.time() - t0, 3)
    timings["cluster_s"] = 0.0
    return global_ids, ecapa_windows, diag
