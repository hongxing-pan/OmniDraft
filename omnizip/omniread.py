"""Joint AV compression by encoder readership.

Each token keeps the distribution of (layer, segment, head, query) positions
that attend to it. Adjacent groups merge by weighted Jensen-Shannon cost.
Audio is a time chain. Video merges only inside a frame. A shared token
budget lets the cheaper modality merge first. Batch non-overlapping merges
approximate the agglomerative information bottleneck; they are not a global
optimum.
"""
import heapq
import time
from collections import defaultdict

import numpy as np

N_Q = 8
LAST = {}


def sample_positions(n, n_q=N_Q):
    if n <= n_q:
        return list(range(n))
    raw = np.linspace(0, n - 1, n_q)
    out = []
    seen = set()
    for x in raw:
        i = int(round(float(x)))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def accumulate_readout(mass, seg, q_used, layer, q, k, cu):
    """Write sampled attention into CPU buffers. q,k are [S, heads, dim]."""
    import torch

    scale = q.shape[-1] ** -0.5
    bounds = [int(x) for x in cu.detach().tolist()]
    for s_i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        n = b - a
        if n <= 0:
            continue
        idx = sample_positions(n, mass.shape[-1])
        sel = torch.tensor(idx, device=q.device, dtype=torch.long)
        qq = q[a:b].index_select(0, sel).permute(1, 0, 2).float()
        kk = k[a:b].permute(1, 2, 0).float()
        prob = torch.softmax(torch.matmul(qq, kk) * scale, dim=-1)
        sl = prob.permute(2, 0, 1).contiguous()
        qn = sl.shape[-1]
        mass[a:b, layer, :, :qn] = sl.to(dtype=torch.float16).cpu()
        seg[a:b, layer] = s_i
        q_used[a:b, layer] = qn


def _scale_mass(mass, q_used, n_heads):
    mass_f = np.asarray(mass, dtype=np.float32)
    used = np.asarray(q_used, dtype=np.float32)
    units = np.maximum(used.sum(axis=1) * float(n_heads), 1.0)
    mass_f /= units[:, None, None, None]
    return mass_f


def pool_audio(mass, seg, lengths):
    """Sum native keys in pairs, matching AvgPool1d(kernel=2, stride=2)."""
    mass = np.asarray(mass)
    seg = np.asarray(seg)
    chunks_m = []
    chunks_s = []
    offset = 0
    for length in lengths:
        length = int(length)
        n = length // 2
        if n:
            pair = mass[offset:offset + n * 2].reshape(n, 2, *mass.shape[1:]).sum(axis=1)
            segs = seg[offset:offset + n * 2:2]
            chunks_m.append(pair)
            chunks_s.append(segs)
        offset += length
    if not chunks_m:
        return mass[:0], seg[:0]
    return np.concatenate(chunks_m, axis=0), np.concatenate(chunks_s, axis=0)


def pool_video(mass, seg, window_index, merge_unit):
    mass = np.asarray(mass)
    seg = np.asarray(seg)
    n = mass.shape[0] // merge_unit
    mass = mass.reshape(n, merge_unit, *mass.shape[1:]).sum(axis=1)
    seg = seg.reshape(n, merge_unit, seg.shape[-1])[:, 0]
    order = np.argsort(np.asarray(window_index))
    return mass[order], seg[order]


def _js_pairs(mass_a, mass_b, seg_a, seg_b, p_a, p_b):
    """Vectorized JS. mass_*: [E, L, ...], seg_*: [E, L]."""
    mass_a = mass_a.reshape(mass_a.shape[0], mass_a.shape[1], -1)
    mass_b = mass_b.reshape(mass_b.shape[0], mass_b.shape[1], -1)
    pt = p_a + p_b
    alpha = np.divide(p_a, pt, out=np.zeros_like(p_a), where=pt > 0)
    pa = p_a.reshape(-1, 1, 1)
    pb = p_b.reshape(-1, 1, 1)
    ra = np.divide(mass_a, pa, out=np.zeros_like(mass_a), where=pa > 0)
    rb = np.divide(mass_b, pb, out=np.zeros_like(mass_b), where=pb > 0)
    m = alpha.reshape(-1, 1, 1) * ra + (1.0 - alpha).reshape(-1, 1, 1) * rb
    m = np.maximum(m, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.sum(np.where(ra > 0, ra * np.log(np.maximum(ra, 1e-12) / m), 0.0), axis=-1)
        kl_b = np.sum(np.where(rb > 0, rb * np.log(np.maximum(rb, 1e-12) / m), 0.0), axis=-1)
    kl_a = np.nan_to_num(kl_a, nan=0.0, posinf=1e6, neginf=0.0)
    kl_b = np.nan_to_num(kl_b, nan=0.0, posinf=1e6, neginf=0.0)
    same = seg_a == seg_b
    layer_a = ra.sum(axis=-1)
    layer_b = rb.sum(axis=-1)
    log_a = np.log(1.0 / np.maximum(alpha, 1e-12))[:, None]
    log_b = np.log(1.0 / np.maximum(1.0 - alpha, 1e-12))[:, None]
    kl_a = np.where(same, kl_a, layer_a * log_a)
    kl_b = np.where(same, kl_b, layer_b * log_b)
    js = alpha * kl_a.sum(axis=-1) + (1.0 - alpha) * kl_b.sum(axis=-1)
    js = np.where(pt > 0, js, 0.0)
    return 0.5 * pt * js


def _parts_from_dense(mass, seg):
    parts = {}
    for li in range(mass.shape[0]):
        parts[(li, int(seg[li]))] = mass[li].copy()
    return parts


def _js_parts(pa, pb, a, b):
    pt = pa + pb
    if pt <= 0:
        return 0.0
    alpha = pa / pt
    keys = set(a) | set(b)
    kl_a = 0.0
    kl_b = 0.0
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if va is None:
            rb = vb / pb if pb > 0 else None
            if rb is not None:
                kl_b += float(rb.sum() * np.log(1.0 / max(1.0 - alpha, 1e-12)))
            continue
        if vb is None:
            ra = va / pa if pa > 0 else None
            if ra is not None:
                kl_a += float(ra.sum() * np.log(1.0 / max(alpha, 1e-12)))
            continue
        ra = va / pa if pa > 0 else np.zeros_like(va)
        rb = vb / pb if pb > 0 else np.zeros_like(vb)
        m = np.maximum(alpha * ra + (1.0 - alpha) * rb, 1e-12)
        if pa > 0:
            kl_a += float(np.sum(np.where(ra > 0, ra * np.log(ra / m), 0.0)))
        if pb > 0:
            kl_b += float(np.sum(np.where(rb > 0, rb * np.log(rb / m), 0.0)))
    return 0.5 * pt * (alpha * kl_a + (1.0 - alpha) * kl_b)


class _Reg:
    __slots__ = ("mass", "seg", "parts", "p", "h", "s", "q", "members", "ri", "rep", "rep_p", "d", "nbrs", "alive", "frame", "ver")

    def __init__(self, mass, seg, h, rep, frame):
        self.mass = mass
        self.seg = seg
        self.parts = None
        self.p = float(mass.sum())
        if self.p <= 0:
            self.p = 1e-8
        self.h = h
        self.s = self.p * h.astype(np.float64)
        self.q = self.p * float(np.dot(h, h))
        self.members = np.array([rep], dtype=np.int32)
        self.ri = 0
        self.rep = rep
        self.rep_p = self.p
        self.d = 0.0
        self.nbrs = set()
        self.alive = True
        self.frame = frame
        self.ver = 0


def _ensure_parts(reg):
    if reg.parts is None:
        reg.parts = _parts_from_dense(reg.mass, reg.seg)
        reg.mass = None
        reg.seg = None
    return reg.parts


def _absorb(dst, src):
    if dst.parts is None and src.parts is None and np.array_equal(dst.seg, src.seg):
        dst.mass = dst.mass + src.mass
    else:
        a = _ensure_parts(dst)
        b = _ensure_parts(src)
        for key, vec in b.items():
            if key in a:
                a[key] = a[key] + vec
            else:
                a[key] = vec
    pt = dst.p + src.p
    if pt > 0:
        dst.h = (dst.p * dst.h + src.p * src.h) / pt
    dst.p = pt
    dst.s = dst.s + src.s
    dst.q = dst.q + src.q
    dst.members = np.concatenate([dst.members, src.members])
    src.members = np.zeros(0, dtype=np.int32)
    if src.rep_p > dst.rep_p or (src.rep_p == dst.rep_p and src.rep < dst.rep):
        dst.rep = src.rep
        dst.rep_p = src.rep_p
    src.alive = False
    if CRITERION == "err":
        dst.d = _region_d(dst)


CRITERION = "usage"
RAWS = []
NORMS = []


def _shift_min(reg, mu):
    h = RAWS[reg.ri][reg.members]
    shift = NORMS[reg.ri][reg.members] + float(np.dot(mu, mu)) - 2.0 * h.dot(mu)
    return float(shift.min()), int(reg.members[int(np.argmin(shift))])


def _region_d(reg):
    """Squared error of the original token nearest the p-weighted mean."""
    p = reg.p
    if p <= 0 or reg.members.size == 0:
        return 0.0
    s = reg.s
    mu = s / p
    disp = reg.q - float(np.dot(s, s)) / p
    mind, _ = _shift_min(reg, mu)
    return float(disp + p * mind)


def _err_delta(a, b):
    p = a.p + b.p
    if p <= 0:
        return 0.0
    s = a.s + b.s
    q = a.q + b.q
    mu = s / p
    disp = q - float(np.dot(s, s)) / p
    best, _ = _shift_min(a, mu)
    if b.members.size:
        other, _ = _shift_min(b, mu)
        best = min(best, other)
    return (disp + p * best) - a.d - b.d


def _edge_cost(a, b, criterion):
    if criterion == "err":
        return _err_delta(a, b)
    if criterion == "mass":
        return a.p + b.p
    if criterion == "feat":
        ha = a.h.astype(np.float64)
        hb = b.h.astype(np.float64)
        na = np.linalg.norm(ha) + 1e-8
        nb = np.linalg.norm(hb) + 1e-8
        return float(1.0 - np.dot(ha, hb) / (na * nb))
    if a.parts is None and b.parts is None:
        return float(_js_pairs(
            a.mass[None], b.mass[None], a.seg[None], b.seg[None],
            np.array([a.p]), np.array([b.p]),
        )[0])
    return _js_parts(a.p, b.p, _ensure_parts(a), _ensure_parts(b))


def _batch_costs(regs, edges, criterion):
    costs = np.empty(len(edges), dtype=np.float64)
    if criterion != "usage":
        for e, (i, j) in enumerate(edges):
            costs[e] = _edge_cost(regs[i], regs[j], criterion)
        return costs
    groups = defaultdict(list)
    for e, (i, j) in enumerate(edges):
        a, b = regs[i], regs[j]
        if a.parts is None and b.parts is None and a.mass.shape == b.mass.shape:
            groups[a.mass.shape].append((e, i, j))
        else:
            costs[e] = _edge_cost(a, b, "usage")
    for shape, items in groups.items():
        ma = np.stack([regs[i].mass for _, i, _ in items])
        mb = np.stack([regs[j].mass for _, _, j in items])
        sa = np.stack([regs[i].seg for _, i, _ in items])
        sb = np.stack([regs[j].seg for _, _, j in items])
        pa = np.array([regs[i].p for _, i, _ in items])
        pb = np.array([regs[j].p for _, _, j in items])
        got = _js_pairs(ma, mb, sa, sb, pa, pb)
        for (e, _, _), c in zip(items, got):
            costs[e] = c
    return costs


def _matching(costs, edges, limit):
    """Non-overlapping merges from the current cheapest cost tier only."""
    if len(edges) == 0 or limit <= 0:
        return []
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        return []
    floor = float(finite.min())
    band = max(1e-8, abs(floor) * 1e-6)
    order = np.argsort(costs, kind="mergesort")
    used = set()
    chosen = []
    for e in order:
        if len(chosen) >= limit:
            break
        if costs[e] > floor + band:
            break
        i, j = edges[e]
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        chosen.append((i, j))
    return chosen


def _apply(regs, chosen):
    if not chosen:
        return
    parent = {}
    for i, j in chosen:
        parent[j] = i
    absorbed = set(parent)
    new_nbrs = defaultdict(set)
    alive_ids = [k for k, r in enumerate(regs) if r.alive]
    for k in alive_ids:
        s = parent.get(k, k)
        for n in regs[k].nbrs:
            sn = parent.get(n, n)
            if sn != s:
                new_nbrs[s].add(sn)
    for i, j in chosen:
        _absorb(regs[i], regs[j])
    for k, r in enumerate(regs):
        if not r.alive:
            r.nbrs = set()
            continue
        r.nbrs = new_nbrs.get(k, set())


def _link(regs, edges):
    for i, j in edges:
        regs[i].nbrs.add(j)
        regs[j].nbrs.add(i)


def _current_edges(regs):
    edges = []
    seen = set()
    for i, r in enumerate(regs):
        if not r.alive:
            continue
        for j in r.nbrs:
            if not regs[j].alive:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    return edges


def _arm_raw(raw):
    arr = np.asarray(raw, dtype=np.float64)
    RAWS.append(arr)
    NORMS.append(np.einsum("ij,ij->i", arr, arr))
    return len(RAWS) - 1


def _build(raw, mass, seg, frames, ri):
    regs = []
    for i in range(raw.shape[0]):
        reg = _Reg(mass[i], seg[i], raw[i].astype(np.float32), i, int(frames[i]) if frames is not None else 0)
        reg.ri = ri
        regs.append(reg)
    return regs


def _push_edge(heap, regs, i, j, criterion):
    if i > j:
        i, j = j, i
    a, b = regs[i], regs[j]
    if (not a.alive) or (not b.alive) or (j not in a.nbrs):
        return
    heapq.heappush(heap, (float(_edge_cost(a, b, criterion)), a.ver, b.ver, i, j))


def _heap_merge(regs, budget, criterion):
    alive = sum(1 for r in regs if r.alive)
    heap = []
    edges = _current_edges(regs)
    if edges and criterion == "usage":
        costs = _batch_costs(regs, edges, criterion)
        for (i, j), cost in zip(edges, costs):
            a, b = regs[i], regs[j]
            heapq.heappush(heap, (float(cost), a.ver, b.ver, i if i < j else j, j if i < j else i))
    else:
        for i, j in edges:
            _push_edge(heap, regs, i, j, criterion)
    while alive > budget and heap:
        _cost, va, vb, i, j = heapq.heappop(heap)
        a, b = regs[i], regs[j]
        if (not a.alive) or (not b.alive) or a.ver != va or b.ver != vb or j not in a.nbrs:
            continue
        nbrs = (a.nbrs | b.nbrs) - {i, j}
        _absorb(a, b)
        a.nbrs = set(nbrs)
        for n in nbrs:
            regs[n].nbrs.discard(j)
            regs[n].nbrs.discard(i)
            regs[n].nbrs.add(i)
        a.ver += 1
        for n in list(a.nbrs):
            _push_edge(heap, regs, i, n, criterion)
        alive -= 1
    return alive


def _compress_graph(regs, init_edges, budget, criterion):
    _link(regs, init_edges)
    return _heap_merge(regs, budget, criterion)


def _audio_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def _video_edges(frames, xs, ys):
    buckets = defaultdict(dict)
    for i, frame in enumerate(frames):
        buckets[int(frame)][(int(ys[i]), int(xs[i]))] = i
    edges = []
    for loc in buckets.values():
        for (y, x), i in loc.items():
            for ny, nx in ((y, x + 1), (y + 1, x)):
                j = loc.get((ny, nx))
                if j is not None:
                    edges.append((i, j))
    return edges


def _center_index(reg):
    p = reg.p
    if p <= 0 or reg.members.size == 0:
        return int(reg.rep)
    _, idx = _shift_min(reg, reg.s / p)
    return idx


def _emit(regs, center=False):
    out = []
    for reg in regs:
        if not reg.alive:
            continue
        rep = _center_index(reg) if center else reg.rep
        out.append({"rep": rep, "h": reg.h, "p": reg.p, "frame": reg.frame})
    return out


def compress_modality(raw, mass, seg, budget, kind, criterion, coords=None, center=False):
    global CRITERION, RAWS, NORMS
    RAWS = []
    NORMS = []
    n = int(raw.shape[0])
    if n == 0:
        return [], 0
    CRITERION = criterion
    ri = _arm_raw(raw)
    frames = None if coords is None else coords[:, 0].astype(np.int64)
    n_frames = 1 if frames is None else len(set(frames.tolist()))
    budget = int(budget)
    if kind == "video":
        budget = max(budget, n_frames)
    budget = max(1 if n else 0, min(n, budget))
    regs = _build(raw, mass, seg, frames if frames is not None else np.zeros(n, dtype=np.int64), ri)
    if kind == "audio":
        edges = _audio_edges(n)
    else:
        edges = _video_edges(frames, coords[:, 1], coords[:, 2])
    _compress_graph(regs, edges, budget, criterion)
    return _emit(regs, center), budget


def compress_shared(audio, video, budget, criterion, center=False):
    """audio/video are (raw, mass, seg, coords or None)."""
    global CRITERION, RAWS, NORMS
    RAWS = []
    NORMS = []
    CRITERION = criterion
    groups = {}
    offset = 0
    combined = []
    edges = []
    for name, pack in (("audio", audio), ("video", video)):
        raw, mass, seg, coords = pack
        if raw is None or len(raw) == 0:
            groups[name] = []
            continue
        frames = np.zeros(len(raw), dtype=np.int64) if coords is None else coords[:, 0].astype(np.int64)
        ri = _arm_raw(raw)
        regs = _build(raw, mass, seg, frames, ri)
        local = _audio_edges(len(regs)) if name == "audio" else _video_edges(frames, coords[:, 1], coords[:, 2])
        _link(regs, local)
        for reg in regs:
            reg.nbrs = {n + offset for n in reg.nbrs}
        groups[name] = regs
        combined.extend(regs)
        offset += len(regs)
    alive = len(combined)
    n_frames = 0
    if video[0] is not None and len(video[0]):
        n_frames = len(set(video[3][:, 0].astype(int).tolist()))
    min_keep = n_frames + (1 if audio[0] is not None and len(audio[0]) else 0)
    budget = int(max(min_keep, min(alive, budget)))
    _heap_merge(combined, budget, criterion)
    return _emit(groups.get("audio", []), center), _emit(groups.get("video", []), center), budget


def omniread_keep_mask(
    input_ids, inputs_embeds, video_grid_thw, audio_token_id, video_token_id,
    keep_ratio, spatial_merge_size, audio_mass=None, audio_seg=None,
    video_mass=None, video_seg=None, read_mode="usage", budget_mode="shared", emit="merge",
    attn_sec=0.0,
):
    from omnizip.omnispan import video_coords
    t0 = time.time()
    ids = input_ids[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    keep = torch_ones(ids.shape[0], device)
    n_a = int(a_pos.numel())
    n_v = int(v_pos.numel())
    raw_a = inputs_embeds[0, a_pos].detach().float().cpu().numpy() if n_a else np.zeros((0, 1), np.float32)
    raw_v = inputs_embeds[0, v_pos].detach().float().cpu().numpy() if n_v else np.zeros((0, 1), np.float32)
    coords = None
    n_frames = 0
    if n_v and video_grid_thw is not None:
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(n_v, tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
    if n_a and (audio_mass is None or len(audio_mass) != n_a):
        raise RuntimeError(f"audio usage {0 if audio_mass is None else len(audio_mass)} != {n_a}")
    if n_v and (video_mass is None or len(video_mass) != n_v):
        raise RuntimeError(f"video usage {0 if video_mass is None else len(video_mass)} != {n_v}")
    if not n_a:
        audio_mass, audio_seg = _uniform(0, 1, 1)
    if not n_v:
        video_mass, video_seg = _uniform(0, 1, 1)
    audio_mass = np.asarray(audio_mass, dtype=np.float32)
    video_mass = np.asarray(video_mass, dtype=np.float32)
    audio_seg = np.asarray(audio_seg, dtype=np.int32)
    video_seg = np.asarray(video_seg, dtype=np.int32)
    criterion = {"usage": "usage", "mass": "mass", "feat": "feat", "err": "err"}[read_mode]
    center = emit == "center"
    clamped = False
    if budget_mode == "fixed":
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a)))) if n_a else 0
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v)))) if n_v else 0
        if n_v and k_v < n_frames:
            clamped = True
            k_v = n_frames
        rec_a, k_a = compress_modality(raw_a, audio_mass, audio_seg, k_a, "audio", criterion, center=center) if n_a else ([], 0)
        rec_v, k_v = compress_modality(raw_v, video_mass, video_seg, k_v, "video", criterion, coords, center) if n_v else ([], 0)
    else:
        k = int(round(keep_ratio * (n_a + n_v)))
        floor = n_frames + (1 if n_a else 0)
        if k < floor:
            clamped = True
            k = floor
        rec_a, rec_v, _ = compress_shared(
            (raw_a, audio_mass, audio_seg, None),
            (raw_v, video_mass, video_seg, coords),
            k, criterion, center,
        )
        k_a, k_v = len(rec_a), len(rec_v)
    _write(inputs_embeds, a_pos, rec_a, keep, emit)
    _write(inputs_embeds, v_pos, rec_v, keep, emit)
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v,
        "read_mode": read_mode, "budget_mode": budget_mode, "emit": emit,
        "video_clamped": clamped, "n_frames": n_frames,
    })
    print(
        "omniread", read_mode, budget_mode, emit,
        "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "sec", LAST["sec"], "attn", LAST["attn_sec"], "clamp", int(clamped),
        flush=True,
    )
    return keep


def _uniform(n, heads, layers):
    if n == 0:
        return np.zeros((0, layers, heads, 1), np.float32), np.zeros((0, layers), np.int32)
    mass = np.zeros((n, layers, heads, 1), np.float32)
    mass[..., 0] = 1.0 / max(layers * heads, 1)
    return mass, np.zeros((n, layers), np.int32)


def torch_ones(n, device):
    import torch
    return torch.ones(n, dtype=torch.bool, device=device)


def _write(inputs_embeds, pos, recs, keep, emit):
    import torch
    if pos.numel() == 0:
        return
    chosen = []
    for rec in recs:
        at = pos[rec["rep"]]
        chosen.append(rec["rep"])
        if emit == "merge":
            src = torch.from_numpy(np.ascontiguousarray(rec["h"])).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds[0, at] = src
    drop = torch.ones(pos.shape[0], dtype=torch.bool)
    if chosen:
        drop[chosen] = False
    keep[pos[drop.to(inputs_embeds.device)]] = False


def _self_check():
    d = 4
    # identical readership merges before orthogonal pair
    mass = np.zeros((4, 1, 1, 2), np.float32)
    mass[0, 0, 0, 0] = 1
    mass[1, 0, 0, 0] = 1
    mass[2, 0, 0, 1] = 1
    mass[3, 0, 0, 1] = 1
    seg = np.zeros((4, 1), np.int32)
    raw = np.eye(4, dtype=np.float32)
    recs, k = compress_modality(raw, mass, seg, 2, "audio", "usage")
    reps = sorted(r["rep"] for r in recs)
    assert k == 2 and reps == [0, 2], reps
    # shared budget spends the cut on the redundant video side
    am = np.zeros((2, 1, 1, 2), np.float32)
    am[0, 0, 0, 0] = 1
    am[1, 0, 0, 1] = 1
    aseg = np.zeros((2, 1), np.int32)
    vm = np.zeros((4, 1, 1, 2), np.float32)
    vm[:, 0, 0, 0] = 1
    vseg = np.zeros((4, 1), np.int32)
    coords = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], np.float64)
    ra, rv, _ = compress_shared(
        (np.ones((2, d), np.float32), am, aseg, None),
        (np.ones((4, d), np.float32), vm, vseg, coords),
        3, "usage",
    )
    assert len(ra) == 2 and len(rv) == 1, (len(ra), len(rv))
    raw4 = np.array([[0.0], [0.0], [5.0], [5.0]], np.float32)
    mass4 = np.ones((4, 1, 1, 1), np.float32)
    recs, k = compress_modality(raw4, mass4, seg, 2, "audio", "err", center=True)
    reps = sorted(r["rep"] for r in recs)
    assert k == 2 and reps[0] in (0, 1) and reps[1] in (2, 3), reps
    # center rep is the original token nearest the weighted mean, not max-p
    raw2 = np.array([[0.0], [0.0], [10.0]], np.float32)
    mass2 = np.ones((3, 1, 1, 1), np.float32)
    mass2[1] = 5
    recs, _ = compress_modality(raw2, mass2, np.zeros((3, 1), np.int32), 1, "audio", "feat", center=True)
    assert recs[0]["rep"] == 0, recs
    print("omniread self-check ok")


if __name__ == "__main__":
    _self_check()
