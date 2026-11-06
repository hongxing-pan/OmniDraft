"""Joint audio-video moment matching with a product kernel.

Time bins come from video frames. Audio tokens are mapped into those bins in
order, so a pair means the two tokens co-occur in time, not that one caused
the other. The kernel is (1+k_a)(1+k_v), with each k a random-feature RBF on
normalized encoder features. Fixed budgets start from audio anchors plus
video facility location, then a few local swaps reduce the joint MMD.
"""
import time

import numpy as np

LAST = {}
RFF = 32
SEED = 0


def _unit(raw):
    h = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(h, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1e-8
    return h / norm


def _features(raw):
    u = _unit(raw)
    n, d = u.shape
    rng = np.random.RandomState(SEED)
    freq = rng.normal(0.0, 1.0, size=(RFF, d))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(RFF,))
    proj = u @ freq.T + phase
    phi = np.sqrt(2.0 / RFF) * np.cos(proj)
    return np.concatenate([np.ones((n, 1)), phi], axis=1)


def _bin_index(bins, n_bins):
    groups = [np.flatnonzero(bins == t) for t in range(n_bins)]
    return groups


def _moment(psi, selected, groups):
    da = psi.shape[1]
    acc = None
    total = 0
    sums = []
    counts = []
    chosen = np.zeros(psi.shape[0], dtype=bool)
    if selected:
        chosen[list(selected)] = True
    for idx in groups:
        take = idx[chosen[idx]] if idx.size else idx
        count = int(take.size)
        counts.append(count)
        if count == 0:
            sums.append(None)
            continue
        sums.append(psi[take].sum(axis=0))
    return sums, counts


def mmd2(psi_a, psi_v, sel_a, sel_v, groups_a, groups_v):
    sum_a, cnt_a = _moment(psi_a, sel_a, groups_a)
    sum_v, cnt_v = _moment(psi_v, sel_v, groups_v)
    total = 0
    acc = None
    for sa, ca, sv, cv in zip(sum_a, cnt_a, sum_v, cnt_v):
        pairs = ca * cv
        if pairs == 0:
            continue
        outer = np.outer(sa, sv)
        acc = outer if acc is None else acc + outer
        total += pairs
    if total == 0 or acc is None:
        return 1e6
    return acc / total, total


def _dist(ref, cur):
    return float(np.sum((ref - cur) ** 2))


def _reference(psi_a, psi_v, groups_a, groups_v):
    mean, _ = mmd2(
        psi_a, psi_v,
        range(psi_a.shape[0]), range(psi_v.shape[0]),
        groups_a, groups_v,
    )
    return mean


def _error(ref, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v):
    cur, _ = mmd2(psi_a, psi_v, sel_a, sel_v, groups_a, groups_v)
    if np.isscalar(cur):
        return float(cur)
    return _dist(ref, cur)


def _local_candidates(token, bins, selected, limit=12):
    t = int(bins[token])
    pool = np.flatnonzero((bins >= t - 1) & (bins <= t + 1))
    out = []
    for i in pool.tolist():
        if i == token or i in selected:
            continue
        out.append(i)
        if len(out) >= limit:
            break
    return out


def _swap_pass(side, selected, bins, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v, ref):
    accepted = 0
    base = _error(ref, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v)
    tokens = list(selected)
    if len(tokens) > 24:
        tokens = tokens[:: max(1, len(tokens) // 24)][:24]
    for token in tokens:
        best_y = None
        best_err = base
        for y in _local_candidates(token, bins, selected):
            trial = set(selected)
            trial.remove(token)
            trial.add(y)
            if side == "a":
                err = _error(ref, psi_a, psi_v, trial, sel_v, groups_a, groups_v)
            else:
                err = _error(ref, psi_a, psi_v, sel_a, trial, groups_a, groups_v)
            if err + 1e-12 < best_err:
                best_err = err
                best_y = y
        if best_y is None:
            continue
        selected.remove(token)
        selected.add(best_y)
        base = best_err
        accepted += 1
    return accepted, base


def _joint_pass(sel_a, sel_v, bins_a, bins_v, psi_a, psi_v, groups_a, groups_v, ref, base):
    audio_tokens = list(sel_a)[:8]
    video_tokens = list(sel_v)[:8]
    best = base
    best_pair = None
    for a in audio_tokens:
        cands_a = _local_candidates(a, bins_a, sel_a, limit=4)
        if not cands_a:
            continue
        for v in video_tokens:
            cands_v = _local_candidates(v, bins_v, sel_v, limit=4)
            if not cands_v:
                continue
            ya, yv = cands_a[0], cands_v[0]
            trial_a = set(sel_a)
            trial_a.remove(a)
            trial_a.add(ya)
            trial_v = set(sel_v)
            trial_v.remove(v)
            trial_v.add(yv)
            err = _error(ref, psi_a, psi_v, trial_a, trial_v, groups_a, groups_v)
            if err + 1e-12 < best:
                best = err
                best_pair = (a, ya, v, yv, err)
    if best_pair is None:
        return 0, base
    a, ya, v, yv, err = best_pair
    sel_a.remove(a)
    sel_a.add(ya)
    sel_v.remove(v)
    sel_v.add(yv)
    return 1, err


def refine(sel_a, sel_v, psi_a, psi_v, bins_a, bins_v, groups_a, groups_v, rounds=2):
    sel_a = set(int(i) for i in sel_a)
    sel_v = set(int(i) for i in sel_v)
    ref = _reference(psi_a, psi_v, groups_a, groups_v)
    swaps = 0
    err = _error(ref, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v)
    for _ in range(rounds):
        got_a, err = _swap_pass("a", sel_a, bins_a, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v, ref)
        got_v, err = _swap_pass("v", sel_v, bins_v, psi_a, psi_v, sel_a, sel_v, groups_a, groups_v, ref)
        got_j, err = _joint_pass(sel_a, sel_v, bins_a, bins_v, psi_a, psi_v, groups_a, groups_v, ref, err)
        swaps += got_a + got_v + got_j
        if got_a + got_v + got_j == 0:
            break
    return sorted(sel_a), sorted(sel_v), swaps, err


def omnimmd_keep_mask(
    input_ids, inputs_embeds, input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, attn_sec=0.0,
):
    import torch
    from omnizip.omnianchor import compress_tokens
    from omnizip.omnifac import select_video
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
    k_a = int(max(1, min(n_a, round(float(keep_ratio) * n_a)))) if n_a else 0
    k_v = int(max(1, min(n_v, round(float(keep_ratio) * n_v)))) if n_v else 0
    clamped = False
    swaps = 0
    err = None
    chosen_a, chosen_v = [], []
    if n_v and video_grid_thw is not None:
        raw_v = emb[v_pos].detach().float().cpu().numpy()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(n_v, tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        if k_v < n_frames:
            clamped = True
            k_v = n_frames
        if video_scores is None or len(video_scores) != n_v:
            vs = np.ones(n_v, dtype=np.float64)
        else:
            vs = np.asarray(video_scores.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1)
        chosen_v = select_video(raw_v, vs, coords, k_v, cross=True, use_attn=True)
        bins_v = coords[:, 0].astype(np.int32)
        remap = {int(t): i for i, t in enumerate(sorted(set(bins_v.tolist())))}
        bins_v = np.array([remap[int(t)] for t in bins_v.tolist()], dtype=np.int32)
        n_bins = int(bins_v.max()) + 1 if n_v else 0
    else:
        raw_v = np.zeros((0, 1))
        bins_v = np.zeros(0, dtype=np.int32)
        n_bins = 0
        chosen_v = []
    if n_a:
        h = emb[a_pos].detach()
        mel_aligned = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            mel_aligned = align_mel(mel, h.shape[0])
        records, k_used = compress_tokens(h, audio_scores, k_a, "audio", acoustic=mel_aligned, anchor_only=True)
        chosen_a = [int(rec["anchor"]) for rec in records]
        if len(chosen_a) > k_a:
            chosen_a = chosen_a[:k_a]
        elif len(chosen_a) < k_a:
            have = set(chosen_a)
            order = list(range(n_a))
            if audio_scores is not None:
                sc = np.asarray(audio_scores.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1)
                if len(sc) == n_a:
                    order = np.argsort(-sc).tolist()
            for i in order:
                if i not in have:
                    chosen_a.append(int(i))
                    have.add(int(i))
                if len(chosen_a) >= k_a:
                    break
        k_a = len(chosen_a)
        raw_a = h.float().cpu().numpy()
        bins_a = np.minimum((np.arange(n_a) * max(n_bins, 1)) // max(n_a, 1), max(n_bins - 1, 0)).astype(np.int32) if n_bins else np.zeros(n_a, dtype=np.int32)
    else:
        raw_a = np.zeros((0, 1))
        bins_a = np.zeros(0, dtype=np.int32)
    if n_a and n_v and n_bins:
        psi_a = _features(raw_a)
        psi_v = _features(raw_v)
        groups_a = _bin_index(bins_a, n_bins)
        groups_v = _bin_index(bins_v, n_bins)
        chosen_a, chosen_v, swaps, err = refine(
            chosen_a, chosen_v, psi_a, psi_v, bins_a, bins_v, groups_a, groups_v, rounds=2,
        )
        k_a = len(chosen_a)
        k_v = len(chosen_v)

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
        "swaps": swaps, "mmd": None if err is None else round(float(err), 6),
        "video_clamped": clamped,
    })
    print(
        "omnimmd", "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "swaps", swaps, "mmd", LAST["mmd"], "sec", LAST["sec"],
        flush=True,
    )
    return keep


def _self_check():
    bark = np.array([1.0, 0.0])
    music = np.array([0.0, 1.0])
    dog = np.array([1.0, 0.0])
    instrument = np.array([0.0, 1.0])
    raw_a = np.stack([bark, bark, music, music, music, bark])
    raw_v = np.stack([dog, instrument])
    bins_a = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    bins_v = np.array([0, 1], dtype=np.int32)
    psi_a = _features(raw_a)
    psi_v = _features(raw_v)
    groups_a = _bin_index(bins_a, 2)
    groups_v = _bin_index(bins_v, 2)
    ref = _reference(psi_a, psi_v, groups_a, groups_v)
    good = _error(ref, psi_a, psi_v, [0, 3], [0, 1], groups_a, groups_v)
    bad = _error(ref, psi_a, psi_v, [2, 5], [0, 1], groups_a, groups_v)
    assert good < bad, (good, bad)
    print("omnimmd self-check ok", round(good, 4), round(bad, 4))


if __name__ == "__main__":
    _self_check()
