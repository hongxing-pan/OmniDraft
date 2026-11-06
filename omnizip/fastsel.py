"""Fast spatiotemporal anchor selection with optional per-slice quotas.

Video runs the same batch facility-location greedy as omnifac.select_video,
with cosine edges on GPU and the whole greedy loop in numba. Audio runs the
OmniAnchor anchor-only merge and records it so merges can be undone
newest-first. Quotas only limit picks made after the first `cap_from` picks;
when every quota equals the unconstrained count the result is unchanged.
"""
import heapq

import numba
import numpy as np
import torch

from omnizip.omnianchor import (
    _absorb, _audio_edges, _build_regions, _energy_features, _link,
    _normalize_scores, _pair_cost,
)


def neighbor_table(coords, cross=True):
    t = coords[:, 0].astype(np.int64)
    x = coords[:, 1].astype(np.int64)
    y = coords[:, 2].astype(np.int64)
    grid = -np.ones((t.max() + 3, y.max() + 3, x.max() + 3), dtype=np.int64)
    grid[t + 1, y + 1, x + 1] = np.arange(t.shape[0])
    dts = (-1, 0, 1) if cross else (0,)
    cols = []
    for dt in dts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cols.append(grid[t + 1 + dt, y + 1 + dy, x + 1 + dx])
    return np.stack(cols, axis=1)


@torch.no_grad()
def cosine_table(raw, nb, device=None):
    h = torch.as_tensor(raw)
    if device is not None:
        h = h.to(device)
    h = h.double()
    norm = torch.linalg.vector_norm(h, dim=1).clamp_min(1e-8)
    hn = h / norm[:, None]
    nb_t = torch.as_tensor(nb, device=hn.device)
    sim = torch.zeros(nb_t.shape, dtype=torch.float64, device=hn.device)
    for o in range(nb_t.shape[1]):
        idx = nb_t[:, o]
        ok = idx >= 0
        sim[ok, o] = (hn[idx[ok]] * hn[ok]).sum(dim=1)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return sim.cpu().numpy()


def to_csr(nb, sim):
    ok = nb >= 0
    indptr = np.zeros(nb.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(ok.sum(axis=1))
    return indptr, nb[ok].astype(np.int32), sim[ok].astype(np.float64)


@numba.njit(cache=True)
def _gains(indptr, indices, data, weight, covered, chosen):
    n = chosen.shape[0]
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        if chosen[j] != 0:
            out[j] = -1.0
            continue
        s = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            d = data[p] - covered[indices[p]]
            if d > 0.0:
                s += weight[indices[p]] * d
        out[j] = s
    return out


@numba.njit(cache=True)
def _cover(indptr, indices, data, covered, j):
    for p in range(indptr[j], indptr[j + 1]):
        if data[p] > covered[indices[p]]:
            covered[indices[p]] = data[p]


@numba.njit(cache=True)
def _batch_greedy(indptr, indices, data, weight, covered, chosen, order, done, budget,
                  slice_of, caps, counts, cap_from):
    n = chosen.shape[0]
    blocked = np.zeros(n, dtype=np.uint8)
    left = budget - done
    while left > 0:
        gain = _gains(indptr, indices, data, weight, covered, chosen)
        rank = np.argsort(-gain, kind="mergesort")
        blocked[:] = 0
        start = done
        for q in range(n):
            j = rank[q]
            if chosen[j] != 0:
                continue
            hit = False
            for p in range(indptr[j], indptr[j + 1]):
                if blocked[indices[p]] != 0:
                    hit = True
                    break
            if hit:
                continue
            if done >= cap_from:
                k = slice_of[j]
                if counts[k] >= caps[k]:
                    continue
                counts[k] += 1
            order[done] = j
            done += 1
            for p in range(indptr[j], indptr[j + 1]):
                blocked[indices[p]] = 1
            if done - start >= left:
                break
        if done == start:
            break
        for q in range(start, done):
            chosen[order[q]] = 1
            _cover(indptr, indices, data, covered, order[q])
        left = budget - done
    return done


def video_graph(raw, coords, cross=True, device=None):
    nb = neighbor_table(coords, cross)
    return to_csr(nb, cosine_table(raw, nb, device))


def video_order(graph, scores, coords, budget, use_attn=True, caps=None, cap_from=None, seed_order=None, token_mul=None):
    """Pick order of omnifac.select_video. caps[k] bounds picks in slice k after cap_from.

    seed_order freezes an already chosen prefix (including the per-frame anchors) and
    only the tokens after it are subject to caps.
    """
    indptr, indices, data = graph
    n = int(indptr.shape[0] - 1)
    frames = coords[:, 0].astype(np.int64)
    first = {}
    for i, f in enumerate(frames.tolist()):
        first.setdefault(f, []).append(i)
    budget = int(max(len(first), min(n, budget)))
    if scores is None or len(scores) != n:
        attn = np.full(n, 1.0 / n)
    else:
        attn = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0)
        total = float(attn.sum())
        attn = attn / total if total > 0 else np.full(n, 1.0 / n)
    weight = 1.0 + n * attn if use_attn else np.ones(n, dtype=np.float64)
    if token_mul is not None:
        weight = weight * np.maximum(np.asarray(token_mul, dtype=np.float64).reshape(-1), 0.0)
    chosen = np.zeros(n, dtype=np.uint8)
    covered = np.zeros(n, dtype=np.float64)
    if seed_order is not None:
        seed_order = np.asarray(seed_order, dtype=np.int64).reshape(-1)
        budget = int(max(seed_order.shape[0], min(n, budget)))
        order = np.full(budget, -1, dtype=np.int64)
        done = 0
        for pick in seed_order.tolist():
            chosen[int(pick)] = 1
            order[done] = int(pick)
            done += 1
            _cover(indptr, indices, data, covered, int(pick))
        n_slices = int(frames.max()) + 1
        if caps is None:
            caps_arr = np.full(n_slices, n, dtype=np.int64)
            cap_from = budget
        else:
            caps_arr = np.asarray(caps, dtype=np.int64)
            cap_from = done
        counts = np.zeros(n_slices, dtype=np.int64)
        done = _batch_greedy(indptr, indices, data, weight, covered, chosen, order, done, budget,
                             frames, caps_arr, counts, cap_from)
        return order[:done]
    order = np.full(budget, -1, dtype=np.int64)
    done = 0
    for ids in first.values():
        ids = np.asarray(ids)
        pick = int(ids[int(np.argmax(attn[ids]))])
        chosen[pick] = 1
        order[done] = pick
        done += 1
        _cover(indptr, indices, data, covered, pick)
    n_slices = int(frames.max()) + 1
    if caps is None:
        caps_arr = np.full(n_slices, n, dtype=np.int64)
        cap_from = budget
    else:
        caps_arr = np.asarray(caps, dtype=np.int64)
        cap_from = int(cap_from)
    counts = np.zeros(n_slices, dtype=np.int64)
    done = _batch_greedy(indptr, indices, data, weight, covered, chosen, order, done, budget,
                         frames, caps_arr, counts, cap_from)
    return order[:done]


def allocate(total, dist, cap):
    """Largest-remainder split of total by dist with part k at most cap[k]."""
    dist = np.maximum(np.asarray(dist, dtype=np.float64), 0.0)
    cap = np.asarray(cap, dtype=np.int64)
    out = np.zeros(cap.shape[0], dtype=np.int64)
    free = cap > 0
    left = int(min(total, cap.sum()))
    while left > 0 and free.any():
        w = np.where(free, dist, 0.0)
        if w.sum() <= 0:
            w = free.astype(np.float64)
        share = left * w / w.sum()
        over = free & (share >= cap - out)
        if over.any():
            left -= int((cap - out)[over].sum())
            out[over] = cap[over]
            free &= ~over
            continue
        base = np.floor(share).astype(np.int64)
        frac = np.where(free, share - base, -1.0)
        base[np.argsort(-frac, kind="mergesort")[:left - int(base.sum())]] += 1
        out += base
        left = 0
    return out


def audio_merge_tree(raw, scores, acoustic, stop_k, snap_k):
    """Anchor-only merge down to stop_k, recording each merge and the snap_k anchors."""
    raw_np = raw.detach().float().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw, dtype=np.float64)
    n = int(raw_np.shape[0])
    stop_k = int(max(1, min(n, stop_k)))
    scores_n = _normalize_scores(None if scores is None else scores, n)
    weights = 1.0 + n * scores_n
    frames = np.zeros(n, dtype=np.int64)
    regs = _build_regions(_energy_features(raw_np), raw_np, weights, scores_n, frames)
    extras = None
    if acoustic is not None:
        ac = acoustic.detach().float().cpu().numpy() if hasattr(acoustic, "detach") else np.asarray(acoustic, dtype=np.float64)
        extras = _build_regions(_energy_features(ac), ac, weights, scores_n, frames)
    edges = _audio_edges(n)
    _link(regs, edges)
    if extras is not None:
        _link(extras, edges)
    merges = []
    prod = [-1] * n
    snap = None
    heap = []

    def push(i, j):
        if i > j:
            i, j = j, i
        a, b = regs[i], regs[j]
        if not a.alive or not b.alive:
            return
        pair = None if extras is None else (extras[i], extras[j])
        heapq.heappush(heap, (_pair_cost(a, b, True, pair), a.ver, b.ver, i, j))

    for i, j in edges:
        push(i, j)
    alive = n
    while alive > stop_k and heap:
        _cost, va, vb, i, j = heapq.heappop(heap)
        a, b = regs[i], regs[j]
        if (not a.alive) or (not b.alive) or a.ver != va or b.ver != vb:
            continue
        if j not in a.nbrs or i not in b.nbrs:
            continue
        if extras is not None:
            _absorb(extras[i], extras[j])
        before_a, before_b = int(a.anchor), int(b.anchor)
        _absorb(a, b)
        merges.append((before_a, before_b, int(a.anchor), prod[i], prod[j]))
        prod[i] = len(merges) - 1
        affected = (a.nbrs | b.nbrs) - {i, j}
        a.nbrs = set(affected)
        b.nbrs.clear()
        for m in affected:
            regs[m].nbrs.discard(j)
            regs[m].nbrs.discard(i)
            regs[m].nbrs.add(i)
        a.ver += 1
        for m in affected:
            push(i, m)
        alive -= 1
        if alive == snap_k and snap is None:
            snap = [int(r.anchor) for r in regs if r.alive]
    final = [int(r.anchor) for r in regs if r.alive]
    roots = [prod[r.idx] for r in regs if r.alive and prod[r.idx] >= 0]
    return {"merges": merges, "final": final, "snap": final if snap is None else snap, "roots": roots}


def audio_undo(tree, target_k, slice_of=None, caps=None, replay=None):
    """Undo merges newest-first until target_k anchors; caps bound added anchors per slice.

    replay is a list of merge ids already undone (a previous round). They are applied
    first, without caps, and the returned trace is those ids plus the new ones.
    """
    merges = tree["merges"]
    anchors = set(tree["final"])
    need = int(target_k) - len(anchors)
    replay = [int(s) for s in (replay or [])]
    heap = [-s for s in tree["roots"]]
    heapq.heapify(heap)
    counts = None if caps is None else np.zeros(len(caps), dtype=np.int64)
    deferred = []
    done_undo = set()
    trace = []

    def undo(s):
        s = int(s)
        if s in done_undo:
            return
        done_undo.add(s)
        trace.append(s)
        before_a, before_b, after, child_a, child_b = merges[s]
        anchors.add(before_b if after == before_a else before_a)
        for c in (child_a, child_b):
            if c >= 0:
                heapq.heappush(heap, -c)

    for s in replay:
        undo(s)
        need -= 1
    heap = [x for x in heap if -int(x) not in done_undo]
    heapq.heapify(heap)

    while need > 0 and heap:
        s = -heapq.heappop(heap)
        if int(s) in done_undo:
            continue
        if counts is not None:
            before_a, before_b, after, _ca, _cb = merges[s]
            k = slice_of[before_b if after == before_a else before_a]
            if counts[k] >= caps[k]:
                deferred.append(s)
                continue
            counts[k] += 1
        undo(s)
        need -= 1
    if need > 0 and deferred:
        for s in deferred:
            heapq.heappush(heap, -s)
        while need > 0 and heap:
            s = -heapq.heappop(heap)
            if int(s) in done_undo:
                continue
            undo(s)
            need -= 1
    tree["undo_trace"] = trace
    return sorted(anchors)
