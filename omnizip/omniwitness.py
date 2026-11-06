"""Video evidence-pair selection.

Audio stays OmniAnchor anchor-only. A video edge joins the same grid cell on
adjacent frames and stores the start token, the end token, and their feature
change. A candidate pair represents that edge only when both endpoints are kept
and the start, end, and change all match. The point mode turns the same edge
values into ordinary single-token gains, so one endpoint can cover the edge.
"""
import time
from collections import defaultdict

import numpy as np

LAST = {}
EPS = 1e-6


def _index(coords):
    loc = {}
    for i, (t, x, y) in enumerate(coords):
        loc[(int(t), int(y), int(x))] = i
    return loc


def _edges(raw, coords, weight):
    loc = _index(coords)
    h = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(h, axis=1)
    norm[norm < 1e-8] = 1e-8
    u = h / norm[:, None]
    edges = []
    for i, (t, x, y) in enumerate(coords):
        j = loc.get((int(t) + 1, int(y), int(x)))
        if j is None:
            continue
        delta = u[j] - u[i]
        edges.append({
            "i": i,
            "j": j,
            "w": 0.5 * (float(weight[i]) + float(weight[j])),
            "delta": delta,
            "dn": float(np.dot(delta, delta)),
        })
    by_cell = {}
    for eidx, edge in enumerate(edges):
        t, x, y = [int(v) for v in coords[edge["i"]]]
        by_cell[(t, int(y), int(x))] = eidx
    for eidx, edge in enumerate(edges):
        t, x, y = [int(v) for v in coords[edge["i"]]]
        cand = []
        for dt in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    fidx = by_cell.get((t + dt, y + dy, x + dx))
                    if fidx is None:
                        continue
                    other = edges[fidx]
                    cos_s = float(np.dot(u[other["i"]], u[edge["i"]]))
                    cos_e = float(np.dot(u[other["j"]], u[edge["j"]]))
                    cos_s = max(0.0, cos_s)
                    cos_e = max(0.0, cos_e)
                    diff = other["delta"] - edge["delta"]
                    num = float(np.dot(diff, diff))
                    change = 1.0 - num / (edge["dn"] + other["dn"] + EPS)
                    change = max(0.0, change)
                    kappa = min(cos_s, cos_e, change)
                    if kappa > 0:
                        cand.append((fidx, kappa))
        edge["cand"] = cand
    covers = defaultdict(list)
    for eidx, edge in enumerate(edges):
        for fidx, kappa in edge["cand"]:
            covers[fidx].append((eidx, kappa))
    return edges, covers, u


def _weights(scores, n):
    if scores is None or len(scores) != n:
        attn = np.full(n, 1.0 / max(n, 1))
    else:
        attn = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0)
        total = float(attn.sum())
        attn = attn / total if total > 0 else np.full(n, 1.0 / n)
    return 1.0 + n * attn, attn


def select_video(raw, scores, coords, budget, mode="pair"):
    n = int(raw.shape[0])
    if n == 0:
        return []
    frames = coords[:, 0].astype(int)
    n_frames = len(set(frames.tolist()))
    budget = int(max(n_frames, min(n, budget)))
    weight, _attn = _weights(scores, n)
    edges, covers, _u = _edges(raw, coords, weight)
    if not edges:
        order = np.argsort(-weight, kind="mergesort")
        return order[:budget].astype(int).tolist()
    chosen = np.zeros(n, dtype=bool)
    covered = np.zeros(len(edges), dtype=np.float64)
    best = np.zeros(len(edges), dtype=np.float64)
    incident = defaultdict(list)
    for eidx, edge in enumerate(edges):
        incident[edge["i"]].append(eidx)
        incident[edge["j"]].append(eidx)
        if edge["cand"]:
            best[eidx] = max(kappa for _f, kappa in edge["cand"])

    def pair_gain(fidx):
        edge = edges[fidx]
        need = []
        if not chosen[edge["i"]]:
            need.append(edge["i"])
        if not chosen[edge["j"]]:
            need.append(edge["j"])
        if not need:
            return 0.0, 0
        gain = 0.0
        for eidx, kappa in covers.get(fidx, ()):
            gain += edges[eidx]["w"] * max(0.0, kappa - covered[eidx])
        return gain, len(need)

    def point_gain(token):
        if chosen[token]:
            return -1.0
        gain = 0.0
        for eidx in incident.get(token, ()):
            gain += edges[eidx]["w"] * max(0.0, best[eidx] - covered[eidx])
        return gain

    def apply_cover(token):
        for eidx in incident.get(token, ()):
            covered[eidx] = max(covered[eidx], best[eidx])

    left = budget
    if mode == "pair":
        while left > 0:
            scores = []
            for fidx in range(len(edges)):
                gain, cost = pair_gain(fidx)
                if cost == 0 or cost > left or gain <= 0:
                    continue
                scores.append((gain / cost, fidx))
            if not scores:
                break
            scores.sort(reverse=True)
            blocked = np.zeros(n, dtype=bool)
            batch = []
            for _score, fidx in scores:
                edge = edges[fidx]
                ends = (edge["i"], edge["j"])
                if any(blocked[t] or chosen[t] and False for t in ends):
                    pass
                if blocked[edge["i"]] or blocked[edge["j"]]:
                    continue
                need = (0 if chosen[edge["i"]] else 1) + (0 if chosen[edge["j"]] else 1)
                if need > left:
                    continue
                batch.append(fidx)
                blocked[edge["i"]] = True
                blocked[edge["j"]] = True
                left -= need
                if left <= 0:
                    break
            if not batch:
                break
            for fidx in batch:
                edge = edges[fidx]
                chosen[edge["i"]] = True
                chosen[edge["j"]] = True
                for eidx, kappa in covers.get(fidx, ()):
                    if kappa > covered[eidx]:
                        covered[eidx] = kappa
        while left > 0:
            gains = np.array([point_gain(i) for i in range(n)])
            j = int(np.argmax(gains))
            if gains[j] < 0:
                break
            chosen[j] = True
            apply_cover(j)
            left -= 1
    else:
        while left > 0:
            gains = np.array([point_gain(i) for i in range(n)])
            j = int(np.argmax(gains))
            if chosen[j]:
                break
            chosen[j] = True
            apply_cover(j)
            left -= 1
    return np.flatnonzero(chosen).astype(int).tolist()


def omniwitness_keep_mask(
    input_ids, inputs_embeds, input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, witness_mode="pair", attn_sec=0.0,
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
        chosen = select_video(vv, scores, coords, k_v, mode=witness_mode)
        k_v = len(chosen)
        drop_unselected(v_pos, chosen)
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v,
        "witness_mode": witness_mode, "video_clamped": clamped,
    })
    print(
        "omniwitness", witness_mode, "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "sec", LAST["sec"], "attn", LAST["attn_sec"], "clamp", int(clamped),
        flush=True,
    )
    return keep


def _self_check():
    raw = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], np.float64)
    coords = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], np.float64)
    scores = np.array([0.4, 0.1, 0.4, 0.1])
    pair = select_video(raw, scores, coords, 2, mode="pair")
    assert set(pair) == {0, 2}, pair
    point = select_video(raw, scores, coords, 2, mode="point")
    assert 0 in point and 2 not in point, point
    print("omniwitness self-check ok", pair, point)


if __name__ == "__main__":
    _self_check()
