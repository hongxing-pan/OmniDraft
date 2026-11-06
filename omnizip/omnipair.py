"""Paired audio-video retention on a shared time axis.

Independent mode maximizes each modality's local coverage on its own.
Joint mode starts there, then swaps equal counts for two rounds to raise
the Hellinger affinity of the two time measures. Shuffle mode uses the same
swaps with the time-bin pairing permuted inside the objective only.
Outputs are original tokens at their original positions.
"""
import time

import numpy as np

LAST = {}


def _unit(raw):
    h = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(h, axis=1)
    norm[norm < 1e-8] = 1e-8
    return h / norm[:, None]


def _bin_weights(score, bins, n_bins):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    score = np.maximum(score, 0.0)
    weights = np.zeros(score.shape[0], dtype=np.float64)
    members = []
    for t in range(n_bins):
        idx = np.flatnonzero(bins == t)
        members.append(idx)
        if idx.size == 0:
            continue
        total = float(score[idx].sum())
        weights[idx] = score[idx] / total if total > 0 else 1.0 / idx.size
    return weights, members


def _audio_covers(raw, bins):
    hn = _unit(raw)
    n_bins = int(bins.max()) + 1 if len(bins) else 0
    by_bin = [np.flatnonzero(bins == t) for t in range(n_bins)]
    covers = []
    for j, t in enumerate(bins.tolist()):
        parts = [by_bin[k] for k in range(max(0, t - 1), min(n_bins, t + 2))]
        nbrs = np.concatenate(parts) if parts else np.array([j], dtype=np.int32)
        sim = hn[nbrs] @ hn[j]
        sim = np.maximum(np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        covers.append((nbrs.astype(np.int32), sim.astype(np.float64)))
    return covers


def _quality(chosen, covers, weights, members):
    n = len(weights)
    best = np.zeros(n, dtype=np.float64)
    for j in chosen:
        nbrs, sim = covers[j]
        best[nbrs] = np.maximum(best[nbrs], sim)
    out = np.zeros(len(members), dtype=np.float64)
    for t, idx in enumerate(members):
        if idx.size:
            out[t] = float(np.sum(weights[idx] * best[idx]))
    return out, best


def _greedy(weights, covers, budget, groups):
    n = len(weights)
    if n == 0:
        return []
    budget = int(max(1, min(n, budget)))
    chosen = np.zeros(n, dtype=bool)
    seeds = []
    for group in groups:
        if group.size == 0:
            continue
        seeds.append(int(group[int(np.argmax(weights[group]))]))
    seeds = sorted(set(seeds), key=lambda i: -weights[i])[:budget]
    covered = np.zeros(n, dtype=np.float64)
    for j in seeds:
        chosen[j] = True
        nbrs, sim = covers[j]
        covered[nbrs] = np.maximum(covered[nbrs], sim)
    left = budget - int(chosen.sum())

    def gains():
        out = np.full(n, -1.0)
        for j in range(n):
            if chosen[j]:
                continue
            nbrs, sim = covers[j]
            out[j] = float(np.sum(weights[nbrs] * np.maximum(0.0, sim - covered[nbrs])))
        return out

    while left > 0:
        gain = gains()
        order = np.argsort(-gain, kind="mergesort")
        blocked = np.zeros(n, dtype=bool)
        batch = []
        for j in order:
            if chosen[j] or gain[j] < 0:
                continue
            nbrs, _sim = covers[int(j)]
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
            nbrs, sim = covers[j]
            covered[nbrs] = np.maximum(covered[nbrs], sim)
        left -= len(batch)
    return np.flatnonzero(chosen).astype(int).tolist()


def _affinity(ua, uv, omega, perm):
    paired = uv if perm is None else uv[perm]
    return float(np.sum(omega * np.sqrt(np.maximum(ua, 0.0) * np.maximum(paired, 0.0))))


def _swap_rounds(sel_a, sel_v, pack_a, pack_v, omega, perm, rounds=2):
    sel_a = set(int(i) for i in sel_a)
    sel_v = set(int(i) for i in sel_v)
    swaps = 0

    def state():
        ua, best_a = _quality(sel_a, pack_a["covers"], pack_a["weights"], pack_a["members"])
        uv, best_v = _quality(sel_v, pack_v["covers"], pack_v["weights"], pack_v["members"])
        return ua, uv, best_a, best_v, _affinity(ua, uv, omega, perm)

    def affected_bins(token, covers, bins):
        nbrs, _sim = covers[token]
        return set(int(b) for b in bins[nbrs].tolist())

    def retarget(which, token, sign, best):
        covers = which["covers"]
        nbrs, sim = covers[token]
        trial = best.copy()
        if sign > 0:
            trial[nbrs] = np.maximum(trial[nbrs], sim)
            return trial
        owners = which["owners"]
        for i, s in zip(nbrs.tolist(), sim.tolist()):
            if best[i] <= s + 1e-12:
                other = -1.0
                for j in owners.get(i, ()):
                    if j == token or j not in (sel_a if which is pack_a else sel_v):
                        continue
                    other = max(other, which["cos_of"][j].get(i, 0.0))
                trial[i] = max(other, 0.0)
        return trial

    for pack in (pack_a, pack_v):
        cos_of = []
        owners = {}
        for j, (nbrs, sim) in enumerate(pack["covers"]):
            cos_of.append({int(i): float(s) for i, s in zip(nbrs.tolist(), sim.tolist())})
            for i in nbrs.tolist():
                owners.setdefault(int(i), []).append(j)
        pack["cos_of"] = cos_of
        pack["owners"] = owners

    for _round in range(rounds):
        for side, selected in (("a", sel_a), ("v", sel_v)):
            pack = pack_a if side == "a" else pack_v
            other_sel = sel_v if side == "a" else sel_a
            ua, uv, best_a, best_v, base = state()
            best = best_a if side == "a" else best_v
            for x in list(selected):
                if x not in selected:
                    continue
                nbrs, _sim = pack["covers"][x]
                local = []
                seen = set()
                for i in nbrs.tolist():
                    for y in pack["owners"].get(int(i), ()):
                        if y in selected or y in seen:
                            continue
                        seen.add(y)
                        local.append(y)
                        if len(local) >= 48:
                            break
                    if len(local) >= 48:
                        break
                best_y = None
                best_j = base
                best_trial = None
                for y in local:
                    trial = retarget(pack, x, -1, best)
                    trial = retarget(pack, y, +1, trial)
                    u_new = np.zeros_like(ua if side == "a" else uv)
                    src = ua if side == "a" else uv
                    u_new[:] = src
                    touched = affected_bins(x, pack["covers"], pack["bins"])
                    touched |= affected_bins(y, pack["covers"], pack["bins"])
                    for t in touched:
                        idx = pack["members"][t]
                        u_new[t] = float(np.sum(pack["weights"][idx] * trial[idx])) if idx.size else 0.0
                    if side == "a":
                        score = _affinity(u_new, uv, omega, perm)
                    else:
                        score = _affinity(ua, u_new, omega, perm)
                    if score > best_j + 1e-12:
                        best_j = score
                        best_y = y
                        best_trial = trial
                if best_y is None:
                    continue
                selected.remove(x)
                selected.add(best_y)
                if side == "a":
                    best_a = best_trial
                    ua, uv, best_a, best_v, base = state()
                    best = best_a
                else:
                    best_v = best_trial
                    ua, uv, best_a, best_v, base = state()
                    best = best_v
                swaps += 1
            other_sel = other_sel
    return sorted(sel_a), sorted(sel_v), swaps


def _mass_score(mass, n):
    if mass is None:
        return np.ones(n, dtype=np.float64)
    score = np.asarray(mass, dtype=np.float64).reshape(n, -1).sum(axis=1)
    if not np.isfinite(score).all() or float(score.sum()) <= 0:
        return np.ones(n, dtype=np.float64)
    return np.maximum(score, 0.0)


def omnipair_keep_mask(
    input_ids, inputs_embeds, video_grid_thw, audio_token_id, video_token_id,
    keep_ratio, spatial_merge_size, audio_mass=None, video_scores=None,
    pair_mode="joint", attn_sec=0.0,
):
    import torch
    from omnizip.omnifac import _covers, _cosine_edges
    from omnizip.omnispan import video_coords
    t0 = time.time()
    ids = input_ids[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    n_a = int(a_pos.numel())
    n_v = int(v_pos.numel())
    n_frames = 0
    coords = None
    if n_v and video_grid_thw is not None:
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(n_v, tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
    k_a = int(max(1, min(n_a, round(float(keep_ratio) * n_a)))) if n_a else 0
    k_v = int(max(1, min(n_v, round(float(keep_ratio) * n_v)))) if n_v else 0
    clamped = False
    if n_v and n_frames and k_v < n_frames:
        clamped = True
        k_v = n_frames
    chosen_a, chosen_v = [], []
    swaps = 0
    score_j = None
    pack_a = None
    pack_v = None
    n_bins = 0
    if n_v and coords is not None:
        raw_v = inputs_embeds[0, v_pos].detach().float().cpu().numpy()
        bins_v = coords[:, 0].astype(np.int32)
        if video_scores is None or len(video_scores) != n_v:
            vscore = np.ones(n_v, dtype=np.float64)
        else:
            vscore = np.maximum(np.asarray(video_scores.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1), 0.0)
        n_use = int(len(set(bins_v.tolist())))
        remap = {int(t): i for i, t in enumerate(sorted(set(bins_v.tolist())))}
        bins_v = np.array([remap[int(t)] for t in bins_v.tolist()], dtype=np.int32)
        weights_v, members_v = _bin_weights(vscore, bins_v, n_use)
        covers_v = _cosine_edges(raw_v, _covers(coords, True))
        chosen_v = _greedy(weights_v, covers_v, k_v, members_v)
        pack_v = {"covers": covers_v, "weights": weights_v, "members": members_v, "bins": bins_v}
        n_bins = n_use
    else:
        pack_v = None
        chosen_v = list(range(n_v))
        n_bins = n_frames
    if n_a and n_bins:
        raw_a = inputs_embeds[0, a_pos].detach().float().cpu().numpy()
        bins_a = np.minimum((np.arange(n_a) * n_bins) // max(n_a, 1), n_bins - 1).astype(np.int32)
        weights_a, members_a = _bin_weights(_mass_score(audio_mass, n_a), bins_a, n_bins)
        covers_a = _audio_covers(raw_a, bins_a)
        chosen_a = _greedy(weights_a, covers_a, k_a, members_a)
        pack_a = {"covers": covers_a, "weights": weights_a, "members": members_a, "bins": bins_a}
    else:
        chosen_a = list(range(n_a))
    if pack_a is not None and pack_v is not None and pair_mode in ("joint", "shuffle") and n_bins:
        omega = np.full(n_bins, 1.0 / n_bins)
        perm = None
        if pair_mode == "shuffle":
            perm = np.random.RandomState(0).permutation(n_bins)
        chosen_a, chosen_v, swaps = _swap_rounds(chosen_a, chosen_v, pack_a, pack_v, omega, perm, rounds=2)
        ua, _ = _quality(chosen_a, pack_a["covers"], pack_a["weights"], pack_a["members"])
        uv, _ = _quality(chosen_v, pack_v["covers"], pack_v["weights"], pack_v["members"])
        if len(ua) == len(uv):
            score_j = _affinity(ua, uv, omega, perm)
    k_a = len(chosen_a) if n_a else 0
    k_v = len(chosen_v) if n_v else 0

    def drop(pos, chosen):
        if pos.numel() == 0:
            return
        mask = torch.ones(pos.shape[0], dtype=torch.bool)
        if chosen:
            mask[list(chosen)] = False
        keep[pos[mask.to(device)]] = False

    drop(a_pos, chosen_a)
    drop(v_pos, chosen_v)
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v,
        "pair_mode": pair_mode, "swaps": swaps, "J": None if score_j is None else round(score_j, 4),
        "video_clamped": clamped, "n_frames": n_frames,
    })
    print(
        "omnipair", pair_mode, "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "swaps", swaps, "J", LAST["J"], "sec", LAST["sec"],
        flush=True,
    )
    return keep
