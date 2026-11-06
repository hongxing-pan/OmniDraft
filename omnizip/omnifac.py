"""Video facility location on a fixed local spatiotemporal graph.

Audio stays OmniAnchor anchor-only. Video keeps original tokens that cover
the largest remaining weighted cosine inside a 3x3 spatial neighborhood and,
unless disabled, the same neighborhood on adjacent frames. Batch selection of
non-overlapping covers is an approximation of serial greedy.
"""
import time
from collections import defaultdict

import numpy as np

LAST = {}


def _video_index(coords):
    loc = {}
    for i, (t, x, y) in enumerate(coords):
        loc[(int(t), int(y), int(x))] = i
    return loc


def _covers(coords, cross):
    loc = _video_index(coords)
    dts = (-1, 0, 1) if cross else (0,)
    covers = []
    for i, (t, x, y) in enumerate(coords):
        t, x, y = int(t), int(x), int(y)
        nbrs = []
        for dt in dts:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    j = loc.get((t + dt, y + dy, x + dx))
                    if j is not None:
                        nbrs.append(j)
        covers.append(np.array(nbrs, dtype=np.int32))
    return covers


def _cosine_edges(raw, covers):
    h = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(h, axis=1)
    norm[norm < 1e-8] = 1e-8
    hn = h / norm[:, None]
    edges = []
    for j, nbrs in enumerate(covers):
        sim = np.nan_to_num(hn[nbrs] @ hn[j], nan=0.0, posinf=0.0, neginf=0.0)
        sim = np.maximum(sim, 0.0)
        edges.append((nbrs, sim.astype(np.float64)))
    return edges


def select_video(raw, scores, coords, budget, cross=True, use_attn=True, weights=None, packed=None):
    n = int(raw.shape[0])
    if n == 0:
        return []
    frames = coords[:, 0].astype(int)
    n_frames = len(set(frames.tolist()))
    budget = int(max(n_frames, min(n, budget)))
    if scores is None or len(scores) != n:
        attn = np.full(n, 1.0 / n)
    else:
        attn = np.asarray(scores, dtype=np.float64).reshape(-1)
        attn = np.maximum(attn, 0)
        total = float(attn.sum())
        attn = attn / total if total > 0 else np.full(n, 1.0 / n)
    base = np.ones(n, dtype=np.float64) if not use_attn else 1.0 + n * attn
    weight = base if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if packed is None:
        packed = _cosine_edges(raw, _covers(coords, cross))
    chosen = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=np.float64)
    by_frame = defaultdict(list)
    for i, frame in enumerate(frames):
        by_frame[int(frame)].append(i)
    for ids in by_frame.values():
        ids = np.array(ids)
        pick = int(ids[int(np.argmax(attn[ids]))])
        chosen[pick] = True
        nbrs, sim = packed[pick]
        covered[nbrs] = np.maximum(covered[nbrs], sim)
    left = budget - int(chosen.sum())

    def gains():
        out = np.full(n, -1.0)
        for j in range(n):
            if chosen[j]:
                continue
            nbrs, sim = packed[j]
            out[j] = float(np.sum(weight[nbrs] * np.maximum(0.0, sim - covered[nbrs])))
        return out

    while left > 0:
        gain = gains()
        order = np.argsort(-gain, kind="mergesort")
        blocked = np.zeros(n, dtype=bool)
        batch = []
        for j in order:
            if chosen[j]:
                continue
            nbrs, _sim = packed[int(j)]
            if blocked[nbrs].any():
                continue
            batch.append(int(j))
            blocked[nbrs] = True
            if len(batch) >= left:
                break
        if not batch:
            break
        for j in batch:
            chosen[j] = True
            nbrs, sim = packed[j]
            covered[nbrs] = np.maximum(covered[nbrs], sim)
        left -= len(batch)
    return np.flatnonzero(chosen).astype(int).tolist()


def omnifac_keep_mask(
    input_ids, inputs_embeds, input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, fac_mode="full", attn_sec=0.0,
):
    import torch
    from omnizip.omnianchor import compress_tokens
    from omnizip.omnispan import align_mel, video_coords
    t0 = time.time()
    ids = input_ids[0]
    emb = inputs_embeds[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    n_a = int(a_pos.numel())
    n_v = int(v_pos.numel())
    k_a, k_v = n_a, n_v
    clamped = False
    cross = fac_mode != "spatial"
    use_attn = fac_mode != "noattn"

    def drop_unselected(pos, chosen):
        if pos.numel() == 0:
            return
        mask = torch.ones(pos.shape[0], dtype=torch.bool)
        if chosen:
            mask[chosen] = False
        keep[pos[mask.to(device)]] = False

    if n_a:
        h = emb[a_pos].detach()
        c = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, h.shape[0])
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        records, k_a = compress_tokens(h, audio_scores, k_a, "audio", acoustic=c, anchor_only=True)
        drop_unselected(a_pos, [rec["anchor"] for rec in records])
    if n_v and video_grid_thw is not None:
        vv = emb[v_pos].detach().float().cpu().numpy()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(vv.shape[0], tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v))))
        if k_v < n_frames:
            clamped = True
            k_v = n_frames
        scores = None if video_scores is None else video_scores.detach().float().cpu().numpy()
        chosen = select_video(vv, scores, coords, k_v, cross=cross, use_attn=use_attn)
        k_v = len(chosen)
        drop_unselected(v_pos, chosen)
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v,
        "fac_mode": fac_mode, "video_clamped": clamped,
    })
    print(
        "omnifac", fac_mode, "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "sec", LAST["sec"], "attn", LAST["attn_sec"], "clamp", int(clamped),
        flush=True,
    )
    return keep


def _self_check():
    raw = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], np.float64)
    coords = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], np.float64)
    scores = np.array([0.7, 0.1, 0.1, 0.1])
    chosen = select_video(raw, scores, coords, 2, cross=True, use_attn=True)
    assert len(chosen) == 2 and 0 in chosen, chosen
    spatial = select_video(raw, scores, coords, 2, cross=False, use_attn=False)
    assert len(spatial) == 2, spatial
    print("omnifac self-check ok", chosen)


if __name__ == "__main__":
    _self_check()
