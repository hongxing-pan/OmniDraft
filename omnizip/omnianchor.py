"""OmniAnchor: training-free LLM-entry compression.

Important tokens stay as their original embeddings when that is the cheaper
representation; other regions become the mean of the original embeddings.
Audio merges only temporal neighbors. Video merges only 4-adjacent regions
inside a frame. Acoustic features affect the audio merge cost only.
"""
import heapq
import time

import numpy as np

_EPS = 1e-6
LAST = {}


def _energy_features(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    energy = float(np.square(centered).sum()) + _EPS
    return centered / energy


def _normalize_scores(scores, n):
    if scores is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    if hasattr(scores, "detach"):
        scores = scores.detach().float().cpu().numpy()
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    if a.shape[0] != n:
        if a.shape[0] > n and a.shape[0] % n == 0:
            a = a.reshape(n, -1).mean(axis=1)
        elif n > 1 and a.shape[0] > 1:
            src = np.linspace(0.0, 1.0, a.shape[0])
            dst = np.linspace(0.0, 1.0, n)
            a = np.interp(dst, src, a)
        else:
            a = np.resize(a, n)
    a = np.maximum(a, 0.0)
    total = float(a.sum())
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return a / total


def _received_attention(q, k, cu_seqlens):
    """Mean attention mass received by each token, within each encoder segment."""
    import torch
    seq, heads, dim = q.shape
    scale = dim ** -0.5
    out = torch.zeros(seq, device=q.device, dtype=torch.float32)
    head_chunk = 4
    seq_chunk = 256
    qf = q.float()
    kf = k.float()
    bounds = cu_seqlens.tolist()
    for seg in range(len(bounds) - 1):
        a = int(bounds[seg])
        b = int(bounds[seg + 1])
        if b - a <= 0:
            continue
        qq = qf[a:b].permute(1, 0, 2)
        kk = kf[a:b].permute(1, 0, 2)
        length = b - a
        acc = torch.zeros(length, device=q.device, dtype=torch.float32)
        for h0 in range(0, heads, head_chunk):
            qh = qq[h0:h0 + head_chunk]
            kh = kk[h0:h0 + head_chunk]
            for i in range(0, length, seq_chunk):
                logits = torch.matmul(qh[:, i:i + seq_chunk], kh.transpose(-1, -2)) * scale
                prob = torch.softmax(logits, dim=-1)
                acc += prob.sum(dim=(0, 1))
                del logits, prob
        out[a:b] = acc / (heads * length)
    return out


class _Region:
    __slots__ = (
        "idx", "members", "count", "sw", "swh", "sw2", "sumf",
        "anchor", "anchor_f", "best_a", "sum_h", "ver", "alive", "nbrs", "frame", "idxs",
    )

    def __init__(self, idx, feat, raw, weight, score, frame):
        self.idx = idx
        self.members = 1
        self.count = 1
        self.sw = float(weight)
        self.swh = weight * feat
        self.sw2 = float(weight * np.dot(feat, feat))
        self.sumf = feat.copy()
        self.anchor = idx
        self.anchor_f = feat.copy()
        self.best_a = float(score)
        self.sum_h = raw.copy()
        self.ver = 0
        self.alive = True
        self.nbrs = set()
        self.frame = frame
        self.idxs = [idx]


def _absorb(dst, src):
    dst.members += src.members
    dst.count += src.count
    dst.sw += src.sw
    dst.swh = dst.swh + src.swh
    dst.sw2 += src.sw2
    dst.sumf = dst.sumf + src.sumf
    dst.sum_h = dst.sum_h + src.sum_h
    dst.idxs.extend(src.idxs)
    if src.best_a > dst.best_a or (src.best_a == dst.best_a and src.anchor < dst.anchor):
        dst.best_a = src.best_a
        dst.anchor = src.anchor
        dst.anchor_f = src.anchor_f
    src.alive = False


def _sse_parts(sw, sw2, swh, y):
    return float(sw2 - 2.0 * np.dot(y, swh) + np.dot(y, y) * sw)


def _dist_parts(sw, sw2, swh, count, sumf, anchor_f, anchor_only):
    d_anchor = _sse_parts(sw, sw2, swh, anchor_f)
    if anchor_only or count <= 1:
        return d_anchor, False
    y_mean = sumf / count
    d_mean = _sse_parts(sw, sw2, swh, y_mean)
    if d_mean < d_anchor:
        return d_mean, True
    return d_anchor, False


def _best_y(reg, anchor_only):
    return _dist_parts(reg.sw, reg.sw2, reg.swh, reg.count, reg.sumf, reg.anchor_f, anchor_only)


def _pair_cost(a, b, anchor_only, extra=None):
    if b.best_a > a.best_a or (b.best_a == a.best_a and b.anchor < a.anchor):
        anchor_f = b.anchor_f
    else:
        anchor_f = a.anchor_f
    d_u, _ = _dist_parts(
        a.sw + b.sw, a.sw2 + b.sw2, a.swh + b.swh, a.count + b.count, a.sumf + b.sumf, anchor_f, anchor_only,
    )
    d_a, _ = _best_y(a, anchor_only)
    d_b, _ = _best_y(b, anchor_only)
    cost = d_u - d_a - d_b
    if extra is None:
        return cost
    return cost + _pair_cost(extra[0], extra[1], anchor_only, None)


def _build_regions(feat, raw, weights, scores, frames):
    regs = []
    for i in range(feat.shape[0]):
        regs.append(_Region(i, feat[i], raw[i], weights[i], scores[i], int(frames[i])))
    return regs


def _merge_until(regs, edges, budget, anchor_only, extras=None, snap_at=None, snap=None):
    heap = []

    def push(i, j):
        if i > j:
            i, j = j, i
        a, b = regs[i], regs[j]
        if not a.alive or not b.alive:
            return
        pair = None if extras is None else (extras[i], extras[j])
        cost = _pair_cost(a, b, anchor_only, pair)
        heapq.heappush(heap, (cost, a.ver, b.ver, i, j))

    for i, j in edges:
        push(i, j)
    alive = sum(1 for r in regs if r.alive)
    while alive > budget and heap:
        cost, va, vb, i, j = heapq.heappop(heap)
        a, b = regs[i], regs[j]
        if (not a.alive) or (not b.alive) or a.ver != va or b.ver != vb:
            continue
        if j not in a.nbrs or i not in b.nbrs:
            continue
        pair = None if extras is None else (extras[i], extras[j])
        if pair is not None:
            _absorb(pair[0], pair[1])
        _absorb(a, b)
        affected = (a.nbrs | b.nbrs) - {i, j}
        a.nbrs = set(affected)
        b.nbrs.clear()
        for n in affected:
            regs[n].nbrs.discard(j)
            regs[n].nbrs.discard(i)
            regs[n].nbrs.add(i)
        a.ver += 1
        for n in affected:
            push(i, n)
        alive -= 1
        if snap is not None and snap_at is not None and alive == snap_at and "anchors" not in snap:
            snap["anchors"] = [int(r.anchor) for r in regs if r.alive]
    return alive


def _audio_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def _video_edges(frames, ys, xs):
    buckets = {}
    for i, frame in enumerate(frames):
        buckets.setdefault(int(frame), []).append(i)
    edges = []
    for ids in buckets.values():
        loc = {(int(ys[i]), int(xs[i])): i for i in ids}
        for (y, x), i in loc.items():
            for ny, nx in ((y, x + 1), (y + 1, x)):
                j = loc.get((ny, nx))
                if j is not None:
                    edges.append((i, j))
                    regs_link = (i, j)
                    _ = regs_link
    return edges


def _link(regs, edges):
    for i, j in edges:
        regs[i].nbrs.add(j)
        regs[j].nbrs.add(i)


def _collect(regs, anchor_only):
    out = []
    for reg in regs:
        if not reg.alive:
            continue
        _, use_mean = _best_y(reg, anchor_only)
        mean = reg.sum_h / reg.count
        out.append({"anchor": reg.anchor, "use_mean": use_mean, "mean": mean, "frame": reg.frame, "idxs": list(reg.idxs)})
    return out


def compress_tokens(raw, scores, budget, kind, coords=None, acoustic=None, anchor_only=False, weights=None):
    """raw: [n, d] torch/numpy. Returns output records and the achieved count."""
    raw_np = raw.detach().float().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw, dtype=np.float64)
    n = int(raw_np.shape[0])
    if n == 0:
        return [], 0
    budget = int(budget)
    if kind == "video" and coords is not None:
        frames = coords[:, 0].astype(int)
        n_frames = int(len(set(frames.tolist())))
        budget = max(budget, n_frames)
    budget = max(1, min(n, budget))
    scores = _normalize_scores(None if scores is None else scores, n)
    if weights is None:
        weights = 1.0 + n * scores
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    feat = _energy_features(raw_np)
    frames = np.zeros(n, dtype=np.int64) if coords is None else coords[:, 0].astype(np.int64)
    regs = _build_regions(feat, raw_np, weights, scores, frames)
    extras = None
    if acoustic is not None:
        ac = acoustic.detach().float().cpu().numpy() if hasattr(acoustic, "detach") else np.asarray(acoustic, dtype=np.float64)
        ac_feat = _energy_features(ac)
        extras = _build_regions(ac_feat, ac, weights, scores, frames)
    if kind == "audio":
        edges = _audio_edges(n)
    else:
        edges = _video_edges(frames, coords[:, 2], coords[:, 1])
    _link(regs, edges)
    if extras is not None:
        _link(extras, edges)
    if n > budget:
        _merge_until(regs, edges, budget, anchor_only, extras)
    return _collect(regs, anchor_only), budget


def audio_anchor_pair(raw, scores, acoustic, full_k, early_k):
    """Anchors at two budgets from one merge. The smaller set is a subset."""
    raw_np = raw.detach().float().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw, dtype=np.float64)
    n = int(raw_np.shape[0])
    if n == 0:
        return [], []
    full_k = int(max(1, min(n, full_k)))
    early_k = int(max(1, min(full_k, early_k)))
    scores = _normalize_scores(None if scores is None else scores, n)
    weights = 1.0 + n * scores
    feat = _energy_features(raw_np)
    frames = np.zeros(n, dtype=np.int64)
    regs = _build_regions(feat, raw_np, weights, scores, frames)
    extras = None
    if acoustic is not None:
        ac = acoustic.detach().float().cpu().numpy() if hasattr(acoustic, "detach") else np.asarray(acoustic, dtype=np.float64)
        extras = _build_regions(_energy_features(ac), ac, weights, scores, frames)
    edges = _audio_edges(n)
    _link(regs, edges)
    if extras is not None:
        _link(extras, edges)
    snap = {}
    if n > early_k:
        _merge_until(regs, edges, early_k, True, extras, snap_at=full_k, snap=snap)
    full_anchors = snap.get("anchors")
    if full_anchors is None:
        full_anchors = [int(r.anchor) for r in regs if r.alive]
    early_anchors = [int(r.anchor) for r in regs if r.alive]
    return early_anchors, full_anchors


def omnanchor_keep_mask(
    input_ids, inputs_embeds, input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, anchor_mode="full", compress="both",
    attn_sec=0.0,
):
    import torch
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
    anchor_only = anchor_mode == "anchor"
    use_attn = anchor_mode != "noattn"
    k_a, k_v = n_a, n_v
    n_mean = 0
    n_raw = 0
    clamped = False

    def apply(pos, records):
        nonlocal n_mean, n_raw
        chosen = []
        for rec in records:
            at = pos[rec["anchor"]]
            chosen.append(rec["anchor"])
            if rec["use_mean"]:
                src = torch.from_numpy(np.ascontiguousarray(rec["mean"])).to(device=device, dtype=inputs_embeds.dtype)
                inputs_embeds[0, at] = src
                n_mean += 1
            else:
                n_raw += 1
        drop = torch.ones(pos.shape[0], dtype=torch.bool)
        if chosen:
            drop[chosen] = False
        keep[pos[drop.to(device)]] = False

    if compress != "video" and n_a:
        h = emb[a_pos].detach()
        c = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, h.shape[0])
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        scores = audio_scores if use_attn else None
        records, k_a = compress_tokens(h, scores, k_a, "audio", acoustic=c, anchor_only=anchor_only)
        apply(a_pos, records)
    if compress != "audio" and n_v and video_grid_thw is not None:
        vv = emb[v_pos].detach()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(vv.shape[0], tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v))))
        if k_v < n_frames:
            clamped = True
            k_v = n_frames
        scores = video_scores if use_attn else None
        records, k_v = compress_tokens(vv, scores, k_v, "video", coords=coords, anchor_only=anchor_only)
        apply(v_pos, records)
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a,
        "k_audio": k_a,
        "n_video": n_v,
        "k_video": k_v,
        "n_raw": n_raw,
        "n_mean": n_mean,
        "anchor_mode": anchor_mode,
        "compress": compress,
        "video_clamped": clamped,
    })
    print(
        "omnanchor", anchor_mode, compress,
        "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "raw", n_raw, "mean", n_mean,
        "sec", LAST["sec"], "attn", LAST["attn_sec"],
        "clamp", int(clamped),
        flush=True,
    )
    return keep


def _self_check():
    raw = np.array([[0.0], [0.0], [10.0], [10.0]])
    scores = np.array([0.01, 0.01, 0.49, 0.49])
    recs, k = compress_tokens(raw, scores, 2, "audio", anchor_only=False)
    assert k == 2 and len(recs) == 2
    recs_a, _ = compress_tokens(raw, scores, 2, "audio", anchor_only=True)
    assert all(not r["use_mean"] for r in recs_a)
    coords = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    vid = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    recs_v, kv = compress_tokens(vid, None, 2, "video", coords=coords)
    frames = sorted(r["frame"] for r in recs_v)
    assert kv == 2 and frames == [0, 1], frames
    print("omnianchor self-check ok", [(r["anchor"], r["use_mean"]) for r in recs])


if __name__ == "__main__":
    _self_check()
