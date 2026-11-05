"""Keep original audio/video tokens by multi-resolution sampling.

The selector runs on encoder outputs. It returns indices only.
Interpolation is used for the audio distortion, not as a model input.
"""
import heapq
import numpy as np
import torch
from numba import njit


@njit
def _minplus(left, right, span, k_offset):
    best = np.empty(span, dtype=np.float64)
    arg = np.empty(span, dtype=np.int32)
    for i in range(span):
        best[i] = np.inf
        arg[i] = 0
    for p in range(left.shape[0]):
        lp = left[p]
        if not np.isfinite(lp):
            continue
        limit = span - (p + k_offset)
        if limit > right.shape[0]:
            limit = right.shape[0]
        for q in range(limit):
            if not np.isfinite(right[q]):
                continue
            k = p + q + k_offset
            v = lp + right[q]
            if v < best[k]:
                best[k] = v
                arg[k] = p
    return best, arg



_EPS = 1e-6


def _centered_energy(x):
    mean = x.mean(dim=0, keepdim=True)
    return (x - mean).pow(2).sum().clamp_min(_EPS)


def _audio_node_cost(h, c, eh, ec, left, right):
    if right <= left + 1:
        return h.new_zeros(())
    i = torch.arange(left + 1, right, device=h.device)
    alpha = (i - left).to(h.dtype) / float(right - left)
    alpha = alpha.unsqueeze(-1)
    h_hat = (1 - alpha) * h[left] + alpha * h[right]
    c_hat = (1 - alpha) * c[left] + alpha * c[right]
    dh = (h[i] - h_hat).pow(2).sum(dim=-1) / eh
    dc = (c[i] - c_hat).pow(2).sum(dim=-1) / ec
    return (dh + dc).sum()


def _build_audio_tree(n):
    nodes = []

    def rec(left, right):
        if right <= left + 1:
            return None
        mid = (left + right) // 2
        idx = len(nodes)
        nodes.append([left, right, mid, None, None])
        nodes[idx][3] = rec(left, mid)
        nodes[idx][4] = rec(mid, right)
        return idx

    root = rec(0, n - 1)
    return nodes, root


def select_audio(h, c, budget):
    """Return kept audio indices. budget is Ka, including the two endpoints."""
    n = h.shape[0]
    if n <= 2 or budget >= n:
        return list(range(n))
    budget = int(max(2, min(n, budget)))
    eh = _centered_energy(h)
    ec = _centered_energy(c)
    nodes, root = _build_audio_tree(n)
    cost = [None] * len(nodes)
    choice = [None] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        left, right, mid, li, ri = nodes[idx]
        span = right - left
        table = [float("inf")] * span
        table[0] = float(_audio_node_cost(h, c, eh, ec, left, right))
        picks = {}
        left_src = cost[li] if li is not None else [0.0]
        right_src = cost[ri] if ri is not None else [0.0]
        if span > 1:
            left_c = np.asarray(left_src, dtype=np.float64)
            right_c = np.asarray(right_src, dtype=np.float64)
            best, arg = _minplus(left_c, right_c, span, 1)
            for k in range(1, span):
                if np.isfinite(best[k]):
                    table[k] = float(best[k])
                    picks[k] = int(arg[k])
        cost[idx] = table
        choice[idx] = picks

    def collect(idx, k, out):
        left, right, mid, li, ri = nodes[idx]
        out.add(left)
        out.add(right)
        if k <= 0:
            return
        p = choice[idx][k]
        q = k - 1 - p
        out.add(mid)
        if li is not None:
            collect(li, p, out)
        if ri is not None:
            collect(ri, q, out)

    if root is None:
        return list(range(n))
    kept = set()
    collect(root, budget - 2, kept)
    return sorted(kept)


def _video_cost_and_rep(v):
    mean = v.mean(dim=0, keepdim=True)
    dist = (v - mean).pow(2).sum(dim=-1)
    rep = int(dist.argmin())
    err = (v - v[rep]).pow(2).sum()
    return rep, err


def _split_region(coords, members):
    if len(members) < 2:
        return None
    pts = coords[members]
    span = pts.max(dim=0).values - pts.min(dim=0).values
    order = torch.argsort(span, descending=True).tolist()
    for dim in order:
        lo = pts[:, dim].min()
        hi = pts[:, dim].max()
        if hi <= lo:
            continue
        mid = (lo + hi) / 2
        left = [m for m, p in zip(members, pts[:, dim]) if p <= mid]
        right = [m for m, p in zip(members, pts[:, dim]) if p > mid]
        if left and right and len(left) < len(members) and len(right) < len(members):
            return left, right
    # Fallback: split the index list in half so the tree can still refine.
    half = len(members) // 2
    return members[:half], members[half:]


def select_video(v, coords, budget):
    n = v.shape[0]
    if n == 0:
        return []
    budget = int(max(1, min(n, budget)))
    if budget >= n:
        return list(range(n))
    nodes = []

    def rec(members):
        idx = len(nodes)
        rep, err = _video_cost_and_rep(v[members])
        nodes.append({"members": members, "rep": members[rep], "err": err.float(), "left": None, "right": None})
        split = _split_region(coords, members)
        if split is None:
            return idx
        left, right = split
        nodes[idx]["left"] = rec(left)
        nodes[idx]["right"] = rec(right)
        return idx

    root = rec(list(range(n)))
    cost = [None] * len(nodes)
    choice = [None] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        m = len(node["members"])
        table = torch.full((m + 1,), float("inf"))
        table[1] = node["err"]
        picks = {}
        li, ri = node["left"], node["right"]
        if li is not None:
            left_c = cost[li].detach().cpu().numpy().astype(np.float64, copy=False)
            right_c = cost[ri].detach().cpu().numpy().astype(np.float64, copy=False)
            best, arg = _minplus(left_c, right_c[1:], m + 1, 1)
            for k in range(2, m + 1):
                if np.isfinite(best[k]):
                    table[k] = float(best[k])
                    picks[k] = int(arg[k])
        cost[idx] = table
        choice[idx] = picks

    def collect(idx, k, out):
        node = nodes[idx]
        if k <= 1 or node["left"] is None or k not in choice[idx]:
            out.add(node["rep"])
            return
        p = choice[idx][k]
        collect(node["left"], p, out)
        collect(node["right"], k - p, out)

    kept = set()
    collect(root, budget, kept)
    return sorted(kept)


def align_mel(mel, n_tokens):
    """Average log-mel frames onto encoder tokens. mel: (n_mels, time)."""
    if mel.dim() != 2:
        raise ValueError(mel.shape)
    time = mel.shape[-1]
    if n_tokens <= 0:
        return mel.new_zeros((0, mel.shape[0]))
    edges = torch.linspace(0, time, n_tokens + 1)
    cols = []
    for i in range(n_tokens):
        a = int(edges[i].item())
        b = max(a + 1, int(edges[i + 1].item()))
        b = min(time, b)
        a = min(a, time - 1)
        cols.append(mel[:, a:b].mean(dim=-1))
    return torch.stack(cols, dim=0)


def video_coords(n, grid_t, grid_h, grid_w, merge):
    nh = max(1, int(grid_h) // merge)
    nw = max(1, int(grid_w) // merge)
    hw = max(1, nh * nw)
    i = torch.arange(n)
    t = i // hw
    rem = i % hw
    y = rem // nw if nw else torch.zeros_like(rem)
    x = rem % nw if nw else torch.zeros_like(rem)
    return torch.stack([t, x, y], dim=1).float()



def _selector_features(h, c):
    eh = _centered_energy(h)
    fh = (h - h.mean(dim=0)) / torch.sqrt(eh)
    if c is None:
        return fh.detach().cpu().numpy().astype(np.float64)
    ec = _centered_energy(c)
    fc = (c - c.mean(dim=0)) / torch.sqrt(ec)
    return torch.cat([fh, fc], dim=-1).detach().cpu().numpy().astype(np.float64)


def ward_segments(h, c, budget):
    """Adjacent Ward merge. Returns contiguous segments covering every token."""
    n = int(h.shape[0])
    budget = int(max(1, min(n, budget)))
    feat = _selector_features(h, c)
    hv = h.detach().float().cpu().numpy().astype(np.float64)
    segs = []
    for i in range(n):
        segs.append({
            "start": i, "end": i, "count": 1,
            "sum_f": feat[i].copy(), "sum_h": hv[i].copy(),
            "prev": i - 1, "next": i + 1 if i + 1 < n else -1,
            "ver": 0, "alive": True,
        })
    heap = []

    def push(i):
        j = segs[i]["next"]
        if i < 0 or j < 0:
            return
        a, b = segs[i], segs[j]
        na, nb = a["count"], b["count"]
        diff = a["sum_f"] / na - b["sum_f"] / nb
        cost = (na * nb) / (na + nb) * float(np.dot(diff, diff))
        heapq.heappush(heap, (cost, a["ver"], b["ver"], i, j))

    for i in range(n - 1):
        push(i)
    alive = n
    while alive > budget and heap:
        cost, va, vb, i, j = heapq.heappop(heap)
        a, b = segs[i], segs[j]
        if (not a["alive"]) or (not b["alive"]) or a["ver"] != va or b["ver"] != vb or a["next"] != j:
            continue
        a["sum_f"] = a["sum_f"] + b["sum_f"]
        a["sum_h"] = a["sum_h"] + b["sum_h"]
        a["count"] += b["count"]
        a["end"] = b["end"]
        a["ver"] += 1
        b["alive"] = False
        nxt = b["next"]
        a["next"] = nxt
        if nxt >= 0:
            segs[nxt]["prev"] = i
            segs[nxt]["ver"] += 1
        prev = a["prev"]
        if prev >= 0:
            segs[prev]["ver"] += 1
            push(prev)
        push(i)
        alive -= 1
    out = []
    i = 0
    while i >= 0:
        s = segs[i]
        if s["alive"]:
            mean = s["sum_h"] / s["count"]
            block = hv[s["start"]:s["end"] + 1]
            rep = s["start"] + int(np.square(block - mean).sum(axis=1).argmin())
            center = (s["start"] + s["end"]) // 2
            out.append({"start": s["start"], "end": s["end"], "rep": rep, "center": center, "mean": mean})
        i = s["next"]
    return out


LAST = {}


def omnispan_keep_mask(input_ids, inputs_embeds, input_features, feature_attention_mask, video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size, audio_mode="span", video_mode="span"):
    import time
    t0 = time.time()
    ids = input_ids[0]
    emb = inputs_embeds[0].float()
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    n_a = int(a_pos.numel())
    k_a = n_a
    n_v = int(v_pos.numel())
    k_v = n_v
    c = None
    h = None
    if n_a:
        h = emb[a_pos].detach().cpu()
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, h.shape[0])
        else:
            c = h
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        if audio_mode == "span":
            local = select_audio(h, c, max(2, k_a) if n_a >= 2 else n_a)
            drop = torch.ones(n_a, dtype=torch.bool)
            drop[local] = False
            keep[a_pos[drop.to(device)]] = False
            k_a = len(local)
        else:
            use_c = None if audio_mode == "ward_mean_h" else c
            segs = ward_segments(h, use_c, k_a)
            chosen = []
            for seg in segs:
                pos = a_pos[seg["center"]]
                chosen.append(seg["center"])
                if audio_mode == "ward_rep":
                    src = emb[a_pos[seg["rep"]]].to(dtype=inputs_embeds.dtype)
                else:
                    src = torch.from_numpy(np.ascontiguousarray(seg["mean"])).to(device=device, dtype=inputs_embeds.dtype)
                inputs_embeds[0, pos] = src
            drop = torch.ones(n_a, dtype=torch.bool)
            drop[chosen] = False
            keep[a_pos[drop.to(device)]] = False
            k_a = len(segs)
    if video_mode != "full" and n_v and video_grid_thw is not None:
        vv = emb[v_pos].detach().cpu()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(vv.shape[0], tt, gh, gw, spatial_merge_size)
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v))))
        local = select_video(vv, coords, k_v)
        drop = torch.ones(n_v, dtype=torch.bool)
        drop[local] = False
        keep[v_pos[drop.to(device)]] = False
        k_v = len(local)
    LAST.clear()
    LAST.update({"sec": round(time.time() - t0, 3), "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v, "audio_mode": audio_mode, "video_mode": video_mode})
    print("omnspan", audio_mode, video_mode, "A", k_a, "/", n_a, "V", k_v, "/", n_v, "sec", LAST["sec"], flush=True)
    return keep
