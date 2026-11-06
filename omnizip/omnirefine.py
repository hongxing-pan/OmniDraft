"""Draft-gradient evidence refinement.

The short draft is the nested prefix of audio anchor-only merging and a fixed
serial video facility order. One frozen full-depth backward on the top-two
answer logits scores omitted tokens. A small remainder of the budget is filled
by that score. Outputs stay original embeddings at original positions.
"""
import heapq
import time

import numpy as np

LAST = {}


def _budgets(k, minimum):
    k = int(k)
    minimum = int(max(1, minimum))
    draft = int(max(minimum, min(k, int(np.ceil(k / 16.0)))))
    refine = int(k // 8)
    core = k - refine
    if core < draft:
        core = draft
        refine = k - core
    return draft, core, refine


def audio_tree(raw, scores, acoustic):
    from omnizip.omnianchor import (
        _audio_edges, _build_regions, _energy_features, _link, _normalize_scores, _pair_cost, _absorb,
    )
    raw_np = raw.detach().float().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw, dtype=np.float64)
    n = int(raw_np.shape[0])
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
    nodes = {i: {"anchor": i, "idxs": [i], "left": None, "right": None, "born": -1} for i in range(n)}
    heap = []

    def push(i, j):
        if i > j:
            i, j = j, i
        a, b = regs[i], regs[j]
        if not a.alive or not b.alive:
            return
        pair = None if extras is None else (extras[i], extras[j])
        cost = _pair_cost(a, b, True, pair)
        heapq.heappush(heap, (cost, a.ver, b.ver, i, j))

    for i, j in edges:
        push(i, j)
    alive = n
    step = 0
    while alive > 1 and heap:
        _cost, va, vb, i, j = heapq.heappop(heap)
        a, b = regs[i], regs[j]
        if (not a.alive) or (not b.alive) or a.ver != va or b.ver != vb:
            continue
        if j not in a.nbrs or i not in b.nbrs:
            continue
        left, right = nodes[i], nodes[j]
        if extras is not None:
            _absorb(extras[i], extras[j])
        _absorb(a, b)
        nodes[i] = {"anchor": int(a.anchor), "idxs": list(a.idxs), "left": left, "right": right, "born": step}
        del nodes[j]
        step += 1
        affected = (a.nbrs | b.nbrs) - {i, j}
        a.nbrs = set(affected)
        b.nbrs.clear()
        for nb in affected:
            regs[nb].nbrs.discard(j)
            regs[nb].nbrs.discard(i)
            regs[nb].nbrs.add(i)
        a.ver += 1
        for nb in affected:
            push(i, nb)
        alive -= 1
    return nodes, weights, raw_np


def _cut(nodes, n, keep):
    """Undo newest merges until `keep` regions remain."""
    cur = dict(nodes)
    while len(cur) < keep:
        parent = max(cur, key=lambda i: cur[i]["born"] if cur[i]["left"] is not None else -1)
        node = cur[parent]
        if node["left"] is None:
            break
        cur.pop(parent)
        cur[parent] = node["left"]
        # right child uses a fresh key that cannot collide with live region ids
        cur[id(node["right"])] = node["right"]
    return list(cur.values())


def video_order(raw, scores, coords, budget):
    """One fixed batch order. Prefixes are not re-solved at a smaller budget."""
    from omnizip.omnifac import _cosine_edges, _covers
    n = int(raw.shape[0])
    frames = coords[:, 0].astype(int)
    n_frames = len(set(frames.tolist()))
    budget = int(max(n_frames, min(n, budget)))
    if scores is None or len(scores) != n:
        attn = np.full(n, 1.0 / n)
    else:
        attn = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0)
        total = float(attn.sum())
        attn = attn / total if total > 0 else np.full(n, 1.0 / n)
    weight = 1.0 + n * attn
    packed = _cosine_edges(raw, _covers(coords, True))
    chosen = []
    chosen_mask = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=np.float64)
    by_frame = {}
    for i, frame in enumerate(frames):
        by_frame.setdefault(int(frame), []).append(i)
    for ids in by_frame.values():
        ids = np.array(ids)
        pick = int(ids[int(np.argmax(attn[ids]))])
        chosen.append(pick)
        chosen_mask[pick] = True
        nbrs, sim = packed[pick]
        covered[nbrs] = np.maximum(covered[nbrs], sim)
    left = budget - len(chosen)

    def gains():
        out = np.full(n, -1.0)
        for j in range(n):
            if chosen_mask[j]:
                continue
            nbrs, sim = packed[j]
            out[j] = float(np.sum(weight[nbrs] * np.maximum(0.0, sim - covered[nbrs])))
        return out

    while left > 0:
        gain = gains()
        blocked = np.zeros(n, dtype=bool)
        batch = []
        for j in np.argsort(-gain, kind="mergesort"):
            if chosen_mask[j]:
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
            chosen.append(j)
            chosen_mask[j] = True
            nbrs, sim = packed[j]
            covered[nbrs] = np.maximum(covered[nbrs], sim)
        left -= len(batch)
    return chosen, packed, weight


def _assign(selected, packed, n):
    rep = np.full(n, -1, dtype=np.int64)
    best = np.full(n, -1.0, dtype=np.float64)
    for j in selected:
        nbrs, sim = packed[int(j)]
        better = sim > best[nbrs]
        idx = nbrs[better]
        rep[idx] = int(j)
        best[idx] = sim[better]
    return rep, best


def _energy(dot, row, rep, weight, mapped):
    valid = mapped & (rep >= 0)
    if not np.any(valid):
        return 0.0
    ii = np.flatnonzero(valid)
    rr = row[ii]
    diff = dot[rr, ii] - dot[rr, rep[ii]]
    return float(np.sum(weight[ii] * diff * diff))


def _edge_index(packed):
    src, dst, sim = [], [], []
    for j, (nbrs, values) in enumerate(packed):
        if len(nbrs) == 0:
            continue
        src.append(np.full(len(nbrs), j, dtype=np.int64))
        dst.append(np.asarray(nbrs, dtype=np.int64))
        sim.append(np.asarray(values, dtype=np.float32))
    if not src:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float32)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(sim)


def _video_gains(selected_mask, best_sim, rep, mapped, src, dst, sim, e_edge, dot_hi, proj, row, weight):
    """One GPU scoring of every legal local edge. No per-candidate sync."""
    import torch
    active = (sim > best_sim[dst]) & mapped[dst] & (rep[dst] >= 0) & (~selected_mask[src])
    if not bool(active.any()):
        return None
    e_cur = (dot_hi[dst] - proj[row[dst], rep.clamp(min=0)[dst]]) ** 2
    delta = weight[dst] * (e_cur - e_edge)
    gain = torch.zeros(selected_mask.shape[0], device=selected_mask.device, dtype=torch.float32)
    gain.index_add_(0, src[active], delta[active])
    gain[selected_mask] = -1
    return gain


def _take_batch(gain, packed, selected_set, room):
    if gain is None or room <= 0:
        return []
    score = gain.detach().float().cpu().numpy()
    blocked = np.zeros(score.shape[0], dtype=bool)
    picked = []
    for j in np.argsort(-score, kind="mergesort"):
        if len(picked) >= room or score[j] <= 0:
            break
        if j in selected_set or blocked[j]:
            continue
        nbrs, _sim = packed[int(j)]
        if blocked[nbrs].any():
            continue
        picked.append(int(j))
        blocked[nbrs] = True
    return picked


def _video_fill(order, core, refine, packed, h, weight, mode, grad_rows, seed):
    n = h.shape[0]
    selected = list(order[:core])
    mapped_np = np.zeros(n, dtype=bool)
    used = 0
    fallback = 0
    cursor = core
    selected_set = set(selected)
    rounds = 0
    if mode != "geom" and grad_rows and refine > 0:
        import torch
        reps = list(order[: len(grad_rows)])
        g = np.stack([grad_rows[i] for i in range(len(reps))], axis=0).astype(np.float32)
        if mode == "shuffle":
            g = g[np.random.default_rng(seed).permutation(len(reps))]
        b, _best = _assign(reps, packed, n)
        mapped_np = b >= 0
        inv = {tok: i for i, tok in enumerate(reps)}
        row = np.array([inv.get(int(x), 0) for x in b], dtype=np.int64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g_t = torch.as_tensor(g, device=device)
        h_t = torch.as_tensor(np.asarray(h, dtype=np.float32), device=device)
        # g[b_i] · h[k] for every draft row and token, computed once
        proj = g_t @ h_t.T
        row_t = torch.as_tensor(row, device=device)
        dot_hi = proj[row_t, torch.arange(n, device=device)]
        src_np, dst_np, sim_np = _edge_index(packed)
        src = torch.as_tensor(src_np, device=device)
        dst = torch.as_tensor(dst_np, device=device)
        sim = torch.as_tensor(sim_np, device=device)
        e_edge = (dot_hi[dst] - proj[row_t[dst], src]) ** 2
        weight_t = torch.as_tensor(np.asarray(weight, dtype=np.float32), device=device)
        mapped = torch.as_tensor(mapped_np, device=device)
        rep_np, best_np = _assign(selected, packed, n)
        for _round in range(2):
            if len(selected) >= core + refine:
                break
            selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
            if selected:
                selected_mask[selected] = True
            rep = torch.as_tensor(rep_np, device=device)
            best_sim = torch.as_tensor(best_np.astype(np.float32), device=device)
            gain = _video_gains(selected_mask, best_sim, rep, mapped, src, dst, sim, e_edge, dot_hi, proj, row_t, weight_t)
            batch = _take_batch(gain, packed, selected_set, core + refine - len(selected))
            rounds += 1
            if not batch:
                break
            for j in batch:
                selected.append(j)
                selected_set.add(j)
                used += 1
                nbrs, values = packed[j]
                better = values > best_np[nbrs]
                idx = nbrs[better]
                rep_np[idx] = j
                best_np[idx] = values[better]
    while len(selected) < core + refine and cursor < len(order):
        while cursor < len(order) and order[cursor] in selected_set:
            cursor += 1
        if cursor >= len(order):
            break
        selected.append(int(order[cursor]))
        selected_set.add(int(order[cursor]))
        cursor += 1
        fallback += 1
    LAST["video_rounds"] = rounds
    return selected, used, fallback, mapped_np


def _audio_select(nodes, n, draft, core, refine, h, weight, mode, grad_by_anchor, seed):
    def regions_at(keep):
        return _cut(nodes, n, keep)

    if mode == "geom":
        chosen = [int(r["anchor"]) for r in regions_at(core + refine)]
        return chosen, 0, refine, np.ones(n, dtype=bool)

    regions = regions_at(core)
    gmap = {}
    anchors = []
    if grad_by_anchor:
        anchors = list(grad_by_anchor.keys())
        g = np.stack([grad_by_anchor[a] for a in anchors], axis=0)
        if mode == "shuffle":
            g = g[np.random.default_rng(seed).permutation(len(anchors))]
        gmap = {a: g[i] for i, a in enumerate(anchors)}
    draft_nodes = regions_at(draft)
    b = np.full(n, -1, dtype=np.int64)
    for node in draft_nodes:
        b[np.array(node["idxs"], dtype=np.int64)] = int(node["anchor"])
    mapped = b >= 0

    # g[b_i] · h[i] and g[b_i] · h[anchor], cached per scored node
    gh = None
    if gmap:
        draft_ids = list(gmap.keys())
        gmat = np.stack([gmap[a] for a in draft_ids], axis=0).astype(np.float64)
        inv = {a: i for i, a in enumerate(draft_ids)}
        grow = np.array([inv.get(int(x), -1) for x in b], dtype=np.int64)
        proj = gmat @ np.asarray(h, dtype=np.float64).T
        gh = (proj, grow)

    def split_drop(node):
        if gh is None or node["left"] is None:
            return 0.0
        proj, grow = gh
        total = 0.0
        parent = int(node["anchor"])
        left_ids = set(node["left"]["idxs"])
        left_anchor = int(node["left"]["anchor"])
        right_anchor = int(node["right"]["anchor"])
        for i in node["idxs"]:
            r = int(grow[i])
            if r < 0:
                continue
            child = left_anchor if i in left_ids else right_anchor
            old = proj[r, i] - proj[r, parent]
            new = proj[r, i] - proj[r, child]
            total += float(weight[i]) * (old * old - new * new)
        return total

    cache = {id(node): split_drop(node) for node in regions if node["left"] is not None}
    used = 0
    fallback = 0
    for _round in range(2):
        room = (core + refine) - len(regions)
        if room <= 0:
            break
        ranked = sorted(
            (node for node in regions if node["left"] is not None and cache.get(id(node), 0.0) > 0),
            key=lambda node: cache[id(node)],
            reverse=True,
        )
        if not ranked:
            break
        taken = 0
        for node in ranked[:room]:
            regions.remove(node)
            regions.append(node["left"])
            regions.append(node["right"])
            cache.pop(id(node), None)
            if node["left"]["left"] is not None:
                cache[id(node["left"])] = split_drop(node["left"])
            if node["right"]["left"] is not None:
                cache[id(node["right"])] = split_drop(node["right"])
            used += 1
            taken += 1
        if taken == 0:
            break
    while len(regions) < core + refine:
        splittable = [node for node in regions if node["left"] is not None]
        if not splittable:
            break
        node = max(splittable, key=lambda item: item["born"])
        regions.remove(node)
        regions.append(node["left"])
        regions.append(node["right"])
        fallback += 1
    chosen = [int(node["anchor"]) for node in regions]
    return chosen, used, fallback, mapped


def _option_scores(logits, option_ids):
    import torch
    valid = [int(i) for i in option_ids if int(i) >= 0]
    if len(valid) < 2:
        return None
    scores = logits[valid]
    order = torch.argsort(scores, descending=True)
    c1 = valid[int(order[0])]
    c2 = valid[int(order[1])]
    return scores[int(order[0])] - scores[int(order[1])], c1, c2


def _draft_gradients(
    text_model, lm_head, input_ids, inputs_embeds, attention_mask, position_ids,
    audio_id, video_id, audio_pos, video_pos, draft_audio, draft_video, option_ids,
):
    import torch
    ids = input_ids[0]
    device = inputs_embeds.device
    media = (ids == audio_id) | (ids == video_id)
    text = (~media).nonzero(as_tuple=True)[0]
    parts = [text]
    if draft_audio:
        parts.append(audio_pos[draft_audio])
    if draft_video:
        parts.append(video_pos[draft_video])
    sel = torch.sort(torch.cat(parts))[0]
    emb = inputs_embeds[:, sel].detach().to(dtype=inputs_embeds.dtype).clone().requires_grad_(True)
    if position_ids is None:
        pos = None
    elif position_ids.dim() == 3:
        pos = position_ids[:, :, sel]
    else:
        pos = position_ids[:, sel]
    mask = None if attention_mask is None else attention_mask[:, sel]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.enable_grad():
        out = text_model(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        logits = lm_head(hidden[:, -1].to(dtype=lm_head.weight.dtype)).float()[0]
        scored = _option_scores(logits, option_ids)
        if scored is None:
            return None
        diff, c1, c2 = scored
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd = time.perf_counter()
        grad_t, = torch.autograd.grad(diff, emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
    back = time.perf_counter()
    grad = grad_t[0].detach().float().cpu().numpy()
    where = {int(p): i for i, p in enumerate(sel.tolist())}
    ga, gv = {}, {}
    for local in draft_audio:
        posi = int(audio_pos[local])
        ga[int(local)] = grad[where[posi]]
    for local in draft_video:
        posi = int(video_pos[local])
        gv[int(local)] = grad[where[posi]]
    return ga, gv, int(c1), int(c2), fwd - t0, back - fwd


def omnirefine_keep_mask(
    text_model, lm_head, input_ids, inputs_embeds, attention_mask, position_ids,
    input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, refine_mode="grad", option_token_ids=None, attn_sec=0.0,
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
    n_a, n_v = int(a_pos.numel()), int(v_pos.numel())
    if n_a == 0 and n_v == 0:
        return keep
    mode = refine_mode if refine_mode in ("grad", "shuffle", "geom") else "grad"
    k_a = int(max(1, min(n_a, round(keep_ratio * n_a)))) if n_a else 0
    k_v = int(max(1, min(n_v, round(keep_ratio * n_v)))) if n_v else 0
    nodes = weights_a = raw_a = None
    acoustic = None
    if n_a:
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            acoustic = align_mel(mel, n_a)
        nodes, weights_a, raw_a = audio_tree(emb[a_pos].detach(), audio_scores, acoustic)
        b_a, c_a, r_a = _budgets(k_a, 1)
    else:
        b_a = c_a = r_a = 0
    order = packed = weights_v = raw_v = coords = None
    clamped = False
    if n_v and video_grid_thw is not None:
        raw_v = emb[v_pos].detach().float().cpu().numpy()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(raw_v.shape[0], tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        if k_v < n_frames:
            clamped = True
            k_v = n_frames
        order, packed, weights_v = video_order(raw_v, None if video_scores is None else video_scores.detach().float().cpu().numpy(), coords, k_v)
        k_v = len(order)
        b_v, c_v, r_v = _budgets(k_v, n_frames)
    else:
        b_v = c_v = r_v = 0
    ga = gv = None
    c1 = c2 = -1
    fwd_sec = back_sec = 0.0
    if mode != "geom":
        try:
            got = _draft_gradients(
                text_model, lm_head, input_ids, inputs_embeds, attention_mask, position_ids,
                audio_token_id, video_token_id, a_pos, v_pos,
                [int(node["anchor"]) for node in _cut(nodes, n_a, b_a)] if n_a else [],
                list(order[:b_v]) if order else [],
                option_token_ids or [],
            )
            if got is None:
                mode = "geom"
            else:
                ga, gv, c1, c2, fwd_sec, back_sec = got
        except Exception as exc:
            print("omnirefine fallback", type(exc).__name__, str(exc)[:180])
            mode = "geom"
            ga = gv = None
    map_a = map_v = 1.0
    used_a = used_v = fb_a = fb_v = 0
    audio_chosen, video_chosen = [], []
    t_score = time.perf_counter()
    if n_a:
        # gradients keyed by local audio index, which is also the anchor id
        audio_chosen, used_a, fb_a, mapped_a = _audio_select(
            nodes, n_a, b_a, c_a, r_a, raw_a, weights_a, mode, ga, 0,
        )
        map_a = float(np.mean(mapped_a))
        drop = torch.ones(n_a, dtype=torch.bool)
        drop[audio_chosen] = False
        keep[a_pos[drop.to(device)]] = False
    if order:
        video_chosen, used_v, fb_v, mapped_v = _video_fill(
            order, c_v, r_v, packed, raw_v, weights_v, mode, [gv[i] for i in order[:b_v]] if gv else [], 0,
        )
        map_v = float(np.mean(mapped_v)) if mode != "geom" else float("nan")
        drop = torch.ones(n_v, dtype=torch.bool)
        drop[video_chosen] = False
        keep[v_pos[drop.to(device)]] = False
    score_sec = time.perf_counter() - t_score
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3), "attn_sec": round(float(attn_sec), 3),
        "mode": mode, "fwd_sec": round(fwd_sec, 3), "back_sec": round(back_sec, 3),
        "score_sec": round(score_sec, 3),
        "n_audio": n_a, "k_audio": len(audio_chosen), "draft_audio": b_a, "core_audio": c_a,
        "n_video": n_v, "k_video": len(video_chosen), "draft_video": b_v, "core_video": c_v,
        "map_audio": None if map_a != map_a else round(map_a, 4),
        "map_video": None if map_v != map_v else round(map_v, 4),
        "used_audio": used_a, "fallback_audio": fb_a, "used_video": used_v, "fallback_video": fb_v,
        "top2": [c1, c2], "video_clamped": clamped,
    })
    extra = float(LAST["fwd_sec"]) + float(LAST["back_sec"]) + float(LAST.get("score_sec") or 0)
    print(
        "omnirefine", LAST["mode"],
        "fwd", LAST["fwd_sec"], "back", LAST["back_sec"], "score", LAST.get("score_sec"),
        "extra", round(extra, 3),
        "A", LAST["k_audio"], "/", n_a, "V", LAST["k_video"], "/", n_v,
        "top", c1, c2, "map", LAST["map_audio"], LAST["map_video"],
    )
    return keep


def _self_check():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(12, 4))
    scores = np.linspace(1, 2, 12)
    nodes, weight, raw_np = audio_tree(raw, scores, None)
    draft, core, refine = _budgets(6, 1)
    chosen, used, fb, mapped = _audio_select(nodes, 12, draft, core, refine, raw_np, weight, "geom", None, 0)
    assert len(set(chosen)) == 6, len(chosen)
    assert mapped.all()
    # a gradient aligned with token 0 should prefer recovering it when omitted
    g = {i: np.zeros(4) for i in range(12)}
    g[int(chosen[0])] = raw_np[0]
    grad_chosen, used, fb, _mapped = _audio_select(nodes, 12, draft, core, refine, raw_np, weight, "grad", g, 0)
    assert len(set(grad_chosen)) == 6
    coords = np.array([[i, 0, 0] for i in range(6)] + [[i, 1, 0] for i in range(6)], dtype=np.int64)
    # 12 tokens, 6 frames, 2 spatial. budget 8
    vv = rng.normal(size=(12, 4))
    order, packed, w = video_order(vv, np.ones(12), coords, 8)
    assert len(order) == 8
    assert len(set(order[:6])) == 6  # one seed per frame
    filled, used, fb, mapped = _video_fill(order, 6, 2, packed, vv, w, "geom", [], 0)
    assert filled[:6] == order[:6]
    assert len(filled) == 8
    print("omnirefine self-check ok", draft, core, refine, "audio", len(chosen), "video", len(order))


if __name__ == "__main__":
    _self_check()
