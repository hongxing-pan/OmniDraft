"""Draft-guided spatiotemporal anchor selection.

The draft is the first part of the spatiotemporal anchor pick order (video)
and a coarser cut of the same anchor merge tree (audio). The draft prefix runs
through the first layers once; a real-question suffix and a null-question
suffix read it from the cache. Their attention difference, pooled per time
slice over both modalities, sets how many of the remaining tokens each slice
receives. Inside a slice the spatiotemporal anchor greedy still decides which
tokens. With lam=0 the final set is exactly the spatiotemporal anchor set.
"""
import time

import numpy as np
import torch

from omnizip.fastsel import allocate, audio_merge_tree, audio_undo, video_graph, video_order
from omnizip.omnispan import align_mel, video_coords

LAST = {}


def _suffix_mask(d, s, dtype, device):
    mask = torch.full((1, 1, s, d + s), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[..., :d] = 0
    mask[0, 0, :, d:] = torch.triu(mask[0, 0, :, d:], diagonal=1)
    return mask


@torch.no_grad()
def probe_attention(thinker, embeds, pos, draft_cols, suffixes, n_layers):
    """Mean attention (heads, suffix tokens) from each suffix to each draft column at layer n_layers."""
    from transformers.cache_utils import DynamicCache
    tm = thinker.model
    layers = tm.layers[:n_layers]
    d = int(draft_cols.numel())
    cache = DynamicCache()
    h = embeds[:, draft_cols]
    p = pos[:, :, draft_cols]
    pe = tm.rotary_emb(h, p)
    cp = torch.arange(d, device=embeds.device)
    for layer in layers:
        h = layer(h, attention_mask=None, position_ids=p, past_key_value=cache,
                  use_cache=True, cache_position=cp, position_embeddings=pe)[0]
    out = []
    for hs, ps in suffixes:
        s = int(hs.shape[1])
        mask = _suffix_mask(d, s, hs.dtype, hs.device)
        pe_s = tm.rotary_emb(hs, ps)
        cps = torch.arange(d, d + s, device=hs.device)
        w = None
        for i, layer in enumerate(layers):
            last = i == len(layers) - 1
            res = layer(hs, attention_mask=mask, position_ids=ps, past_key_value=cache,
                        output_attentions=last, use_cache=True, cache_position=cps, position_embeddings=pe_s)
            hs = res[0]
            if last:
                w = res[1]
        out.append(w[0, :, :, :d].float().mean(dim=(0, 1)))
        cache.crop(d)
    return out


@torch.no_grad()
def probe_reference(thinker, embeds, pos, draft_cols, suffix, n_layers):
    """Same quantity as probe_attention, from one uncached forward of draft + suffix."""
    tm = thinker.model
    layers = tm.layers[:n_layers]
    hs, ps = suffix
    d = int(draft_cols.numel())
    h = torch.cat([embeds[:, draft_cols], hs], dim=1)
    p = torch.cat([pos[:, :, draft_cols], ps], dim=2)
    n = int(h.shape[1])
    mask = torch.triu(torch.full((n, n), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device), diagonal=1)[None, None]
    pe = tm.rotary_emb(h, p)
    cp = torch.arange(n, device=h.device)
    w = None
    for i, layer in enumerate(layers):
        last = i == len(layers) - 1
        res = layer(h, attention_mask=mask, position_ids=p, past_key_value=None,
                    output_attentions=last, use_cache=False, cache_position=cp, position_embeddings=pe)
        h = res[0]
        if last:
            w = res[1]
    return w[0, :, d:, :d].float().mean(dim=(0, 1))


def _null_suffix(thinker, ids, pos, start, cfg):
    suffix = ids[start:].tolist()
    special = set(cfg.get("special_ids", []))
    m = 0
    while m < len(suffix) and suffix[m] in special:
        m += 1
    im_end = int(cfg.get("im_end_id", -1))
    ends = [i for i, t in enumerate(suffix) if t == im_end]
    k = ends[-1] if ends else len(suffix)
    null = suffix[:m] + list(cfg["null_question_ids"]) + suffix[k:]
    t = torch.tensor(null, device=ids.device)
    emb = thinker.get_input_embeddings()(t)[None]
    p0 = pos[:, :, start:start + 1]
    p = p0 + torch.arange(len(null), device=ids.device).view(1, 1, -1)
    return emb, p


def _slice_relevance(excess, kinds, slices, n_slices):
    parts = []
    for kind in (0, 1):
        sel = kinds == kind
        mass = np.bincount(slices[sel], weights=excess[sel], minlength=n_slices)
        if mass.sum() > 0:
            parts.append(mass / mass.sum())
    if not parts:
        return None
    rel = np.mean(parts, axis=0)
    pad = np.pad(rel, 1, mode="edge")
    rel = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return rel / rel.sum()


def omnidraft_keep_mask(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, cfg=None, attn_sec=0.0,
):
    cfg = cfg or {}
    rho = float(cfg.get("draft_ratio", 0.5))
    lam_max = float(cfg.get("lam", 0.0))
    n_layers = int(cfg.get("probe_layer", 12))
    kappa = float(cfg.get("kappa", 1e9))
    g0 = float(cfg.get("g0", 0.0))
    exercise = bool(cfg.get("exercise", False))
    t0 = time.perf_counter()
    ids = input_ids[0]
    emb = inputs_embeds[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    n_a, n_v = int(a_pos.numel()), int(v_pos.numel())
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    tpos = position_ids[0, 0].detach().cpu().numpy()

    coords = frames = order = graph = vscores = None
    n_slices = 1
    k_v = b_v = 0
    if n_v and video_grid_thw is not None:
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(n_v, tt, gh, gw, spatial_merge_size).numpy()
        frames = coords[:, 0].astype(np.int64)
        n_frames = int(len(set(frames.tolist())))
        n_slices = int(frames.max()) + 1
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v))))
        k_v = max(k_v, n_frames)
        vscores = None if video_scores is None else video_scores.detach().float().cpu().numpy()
        graph = video_graph(emb[v_pos].detach(), coords, True, device)
        order = video_order(graph, vscores, coords, k_v)
        k_v = int(order.shape[0])
        b_v = int(min(k_v, max(n_frames, round(rho * k_v))))
    tree = None
    a_slice = None
    k_a = b_a = 0
    if n_a:
        h = emb[a_pos].detach()
        c = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, n_a)
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        b_a = int(max(1, round(rho * k_a)))
        tree = audio_merge_tree(h, audio_scores, c, b_a, k_a)
        if frames is not None:
            vt = tpos[v_pos.cpu().numpy()]
            starts = np.array([vt[np.argmax(frames == k)] for k in range(n_slices)])
            a_slice = np.clip(np.searchsorted(starts, tpos[a_pos.cpu().numpy()], side="right") - 1, 0, n_slices - 1)
        else:
            a_slice = np.zeros(n_a, dtype=np.int64)
    t_sel = time.perf_counter()

    rel = rel_v = rel_a = None
    g = 0.0
    lam = 0.0
    saved_draft = None
    saved_start = None
    probe_sec = 0.0
    if lam_max > 0 and order is not None and tree is not None:
        a_np, v_np = a_pos.cpu().numpy(), v_pos.cpu().numpy()
        start = int(max(a_np.max(), v_np.max())) + 1
        media = np.zeros(ids.shape[0], dtype=bool)
        media[a_np] = True
        media[v_np] = True
        draft = ~media
        draft[start:] = False
        draft[v_np[order[:b_v]]] = True
        draft[a_np[tree["final"]]] = True
        draft_cols = torch.from_numpy(np.flatnonzero(draft)).to(device)
        saved_draft = draft_cols.detach().cpu().tolist()
        saved_start = start
        real = (inputs_embeds[:, start:], position_ids[:, :, start:])
        use_null = int(cfg.get("null_sub", 1))
        suffixes = [real]
        if use_null:
            suffixes.append(_null_suffix(thinker, ids, position_ids, start, cfg))
        tp = time.perf_counter()
        probed = probe_attention(thinker, inputs_embeds, position_ids, draft_cols, suffixes, n_layers)
        a_q = probed[0]
        use_prior = cfg.get("signal") == "prior"
        if use_prior and not use_null:
            raise RuntimeError("prior correction needs the null suffix")
        if use_prior:
            excess = None
        elif use_null:
            excess = torch.clamp(a_q - probed[1], min=0).cpu().numpy()
        else:
            excess = a_q.cpu().numpy()
        probe_sec = time.perf_counter() - tp
        if cfg.get("check") and use_null:
            ref_q = probe_reference(thinker, inputs_embeds, position_ids, draft_cols, real, n_layers)
            ref_0 = probe_reference(thinker, inputs_embeds, position_ids, draft_cols, suffixes[1], n_layers)
            for name, x, y in (("real", a_q, ref_q), ("null", probed[1], ref_0)):
                err = float((x - y).abs().max() / y.abs().max().clamp_min(1e-12))
                corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
                print("omnidraft probe_check", name, "rel_max_err", round(err, 5), "corr", round(corr, 5), flush=True)
        cols = draft_cols.cpu().numpy()
        kind = np.full(ids.shape[0], -1, dtype=np.int64)
        kind[a_np] = 0
        kind[v_np] = 1
        slc = np.zeros(ids.shape[0], dtype=np.int64)
        slc[a_np] = a_slice
        slc[v_np] = frames
        on = kind[cols] >= 0
        if use_prior:
            def _dist(x):
                x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
                z = float(x.sum())
                if z <= 1e-12:
                    return np.full_like(x, 1.0 / max(1, x.size))
                return x / z
            post = _dist(a_q.float().cpu().numpy()[on])
            prior = _dist(probed[1].float().cpu().numpy()[on])
            eps = 1e-8
            lift = post * np.log((post + eps) / (prior + eps))
            excess = np.zeros(cols.shape[0], dtype=np.float64)
            excess[on] = np.clip(lift, 0, None)
        use_cross = int(cfg.get("cross", 1))
        if use_cross:
            rel = _slice_relevance(excess[on], kind[cols][on], slc[cols][on], n_slices)
            rel_v = rel_a = rel
        else:
            rel_v = _slice_relevance(excess[on], np.where(kind[cols][on] == 1, 1, -1), slc[cols][on], n_slices)
            rel_a = _slice_relevance(excess[on], np.where(kind[cols][on] == 0, 0, -1), slc[cols][on], n_slices)
            rel = rel_v if rel_v is not None else rel_a
        if rel is not None:
            ent = -float(np.sum(rel[rel > 0] * np.log(rel[rel > 0])))
            g = 1.0 - ent / np.log(max(2, n_slices))
            if int(cfg.get("gate", 0)):
                lam = min(lam_max, max(0.0, kappa * (g - g0)))
            else:
                lam = lam_max

    def _mul(dist):
        dist = np.asarray(dist, dtype=np.float64)
        dist = dist / max(float(dist.mean()), 1e-12)
        return (1.0 - lam) + lam * dist

    use_weight = cfg.get("signal", "quota") == "weight"
    final_v = order
    if order is not None and use_weight and lam > 0 and rel_v is not None:
        final_v = video_order(graph, vscores, coords, k_v, weight_mul=_mul(rel_v)[frames])
    elif order is not None and (lam > 0 or exercise):
        K = k_v - b_v
        default = np.bincount(frames[order[b_v:]], minlength=n_slices)
        room = np.bincount(frames, minlength=n_slices) - np.bincount(frames[order[:b_v]], minlength=n_slices)
        dist = default / max(1, K)
        if lam > 0 and rel_v is not None:
            dist = (1 - lam) * dist + lam * rel_v
        caps = allocate(K, dist, room)
        final_v = video_order(graph, vscores, coords, k_v, caps=caps, cap_from=b_v)
    final_a = None
    if tree is not None and use_weight and lam > 0 and rel_a is not None:
        weighted = audio_merge_tree(h, audio_scores, c, k_a, k_a, weight_mul=_mul(rel_a)[a_slice])
        final_a = weighted["final"]
    elif tree is not None:
        final_a = tree["snap"]
        if lam > 0 or exercise:
            Ka = k_a - b_a
            added = sorted(set(tree["snap"]) - set(tree["final"]))
            default = np.bincount(a_slice[added], minlength=n_slices)
            room = np.bincount(a_slice, minlength=n_slices) - np.bincount(a_slice[tree["final"]], minlength=n_slices)
            dist = default / max(1, Ka)
            if lam > 0 and rel_a is not None:
                dist = (1 - lam) * dist + lam * rel_a
            final_a = audio_undo(tree, k_a, a_slice, allocate(Ka, dist, room))

    if final_a is not None:
        drop = torch.ones(n_a, dtype=torch.bool)
        drop[torch.as_tensor(sorted(final_a), dtype=torch.long)] = False
        keep[a_pos[drop.to(device)]] = False
    if final_v is not None:
        drop = torch.ones(n_v, dtype=torch.bool)
        drop[torch.as_tensor(np.sort(final_v), dtype=torch.long)] = False
        keep[v_pos[drop.to(device)]] = False

    if cfg.get("check"):
        from omnizip.omnifac import omnifac_keep_mask
        ref = omnifac_keep_mask(
            input_ids=input_ids, inputs_embeds=inputs_embeds, input_features=input_features,
            feature_attention_mask=feature_attention_mask, video_grid_thw=video_grid_thw,
            audio_token_id=audio_token_id, video_token_id=video_token_id, keep_ratio=keep_ratio,
            spatial_merge_size=spatial_merge_size, audio_scores=audio_scores, video_scores=video_scores,
        )
        print("omnidraft check", "lam", round(lam, 4), "same_as_omnifac", bool(torch.equal(ref, keep)), flush=True)

    moved_v = 0 if final_v is None or order is None else int(len(set(final_v.tolist()) - set(order.tolist())))
    moved_a = 0 if final_a is None else int(len(set(final_a) - set(tree["snap"])))
    LAST.clear()
    LAST.update({
        "sec": round(time.perf_counter() - t0, 3),
        "select_sec": round(t_sel - t0, 3),
        "probe_sec": round(probe_sec, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "draft_audio": b_a,
        "n_video": n_v, "k_video": k_v, "draft_video": b_v,
        "g": round(g, 4), "lam": round(lam, 4), "moved_video": moved_v, "moved_audio": moved_a,
    })
    print("omnidraft", " ".join(f"{k}={v}" for k, v in LAST.items()), flush=True)
    LAST["draft_pos"] = saved_draft
    LAST["question_at"] = saved_start
    return keep


def _run_layers(text_model, hidden, position_ids, layers, cache, past_len, mask=None):
    seq = hidden.shape[1]
    cache_position = torch.arange(past_len, past_len + seq, device=hidden.device)
    position_embeddings = text_model.rotary_emb(hidden, position_ids)
    for layer in layers:
        hidden = layer(
            hidden, attention_mask=mask, position_ids=position_ids, past_key_value=cache,
            output_attentions=False, use_cache=True, cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    return hidden


def prefill_reuse4(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_scores, video_scores, keep_ratio, cfg,
):
    """Same keep set as OmniDraft. Resume from the layer-4 state instead of rerunning all 36 layers."""
    from transformers.cache_utils import DynamicCache
    from omnizip.omnilate import _catchup

    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    keep = omnidraft_keep_mask(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, thinker.config.audio_token_id, thinker.config.video_token_id,
        keep_ratio, int(thinker.visual.spatial_merge_size), audio_scores, video_scores, cfg,
    )
    draft_pos = LAST.get("draft_pos")
    if not draft_pos:
        raise RuntimeError("reuse4 needs the draft positions from the probe")
    text_model = thinker.model
    layers = list(text_model.layers)
    device = inputs_embeds.device
    full_pos = torch.nonzero(keep, as_tuple=True)[0]
    draft_index = torch.tensor(draft_pos, device=device)
    start = int(LAST.get("question_at"))
    # Question is the kept tail after the last media token. Draft positions already include the system text.
    question_pos = torch.arange(start, keep.numel(), device=device)
    question_pos = question_pos[keep[question_pos]]
    draft_set = set(draft_pos)
    new_pos = torch.tensor([int(p) for p in full_pos.tolist() if int(p) not in draft_set and int(p) < start], device=device)

    early = DynamicCache()
    h4_draft = _run_layers(
        text_model, inputs_embeds[:, draft_index], position_ids[:, :, draft_index],
        layers[:4], early, 0,
    )
    if question_pos.numel():
        q = int(question_pos.numel())
        mask = _suffix_mask(len(draft_pos), q, inputs_embeds.dtype, device)
        h4_question = _run_layers(
            text_model, inputs_embeds[:, question_pos], position_ids[:, :, question_pos],
            layers[:4], early, len(draft_pos), mask,
        )
    else:
        h4_question = None

    hidden = inputs_embeds[:, full_pos].clone()
    draft_of = {p: i for i, p in enumerate(draft_pos)}
    question_of = {int(p): i for i, p in enumerate(question_pos.tolist())}
    if new_pos.numel():
        caught = _catchup(inputs_embeds[:, new_pos], inputs_embeds[:, draft_index], h4_draft)
        new_of = {int(p): i for i, p in enumerate(new_pos.tolist())}
    else:
        caught = None
        new_of = {}
    for i, p in enumerate(full_pos.tolist()):
        p = int(p)
        if p in draft_of:
            hidden[0, i] = h4_draft[0, draft_of[p]]
        elif p in question_of:
            hidden[0, i] = h4_question[0, question_of[p]]
        else:
            hidden[0, i] = caught[0, new_of[p]]

    late = DynamicCache()
    hidden = _run_layers(text_model, hidden, position_ids[:, :, full_pos], layers[4:], late, 0)
    hidden = text_model.norm(hidden)
    thinker._reuse_state = {
        "early": early,
        "late": late,
        "early_len": len(draft_pos) + int(question_pos.numel()),
        "full_len": int(full_pos.numel()),
        "next_pos": position_ids[:, :, full_pos][:, :, -1:] + 1,
    }
    LAST["exec"] = "reuse4"
    LAST["reuse_new"] = int(new_pos.numel())
    print("reuse4", "draft", len(draft_pos), "full", int(full_pos.numel()), "new", int(new_pos.numel()), flush=True)
    LAST.pop("draft_pos", None)
    LAST.pop("question_at", None)
    return hidden


def _index_mask(q_idx, k_idx, dtype, device):
    allow = k_idx[None, :] <= q_idx[:, None]
    mask = torch.full((1, 1, q_idx.numel(), k_idx.numel()), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[0, 0] = torch.where(allow, torch.zeros((), dtype=dtype, device=device), mask[0, 0])
    return mask


def _empty_cache():
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    cache.key_cache = []
    cache.value_cache = []
    cache._seen_tokens = 0
    return cache


def _seed_layer(cache, layer_idx, key, value):
    while len(cache.key_cache) <= layer_idx:
        cache.key_cache.append(torch.tensor([], device=key.device))
        cache.value_cache.append(torch.tensor([], device=key.device))
    cache.key_cache[layer_idx] = key
    cache.value_cache[layer_idx] = value


def prefill_splice(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_scores, video_scores, keep_ratio, cfg,
):
    """Same keep set as OmniDraft. Draft tokens before layer L never see newly added tokens."""
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    keep = omnidraft_keep_mask(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, thinker.config.audio_token_id, thinker.config.video_token_id,
        keep_ratio, int(thinker.visual.spatial_merge_size), audio_scores, video_scores, cfg,
    )
    draft_pos = LAST.get("draft_pos")
    if not draft_pos:
        raise RuntimeError("splice needs the draft positions from the probe")
    text_model = thinker.model
    layers = list(text_model.layers)
    n_layers = len(layers)
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    L = max(0, min(n_layers, int(cfg.get("splice_l", 12))))
    repair_ratio = float(cfg.get("splice_r", 0.0))
    full_pos = torch.nonzero(keep, as_tuple=True)[0]
    draft_index = torch.tensor(draft_pos, device=device, dtype=torch.long)
    start = int(LAST["question_at"])
    draft_set = set(draft_pos)
    missing = [int(p) for p in draft_pos if not bool(keep[p])]
    if missing:
        raise RuntimeError(f"draft token {missing[0]} is outside the final keep set")
    pos_list = [int(p) for p in full_pos.tolist()]
    loc = {p: i for i, p in enumerate(pos_list)}
    d_pos = [p for p in pos_list if p in draft_set]
    q_pos = [p for p in pos_list if p not in draft_set]
    if d_pos != draft_pos:
        raise RuntimeError("draft positions are not in original order")
    t0 = time.perf_counter()
    hidden_full = None
    early = _empty_cache()
    if L == 0:
        late_in = inputs_embeds[:, full_pos]
    else:
        d_idx = torch.tensor(d_pos, device=device)
        draft_cache = _empty_cache()
        h_d = _run_layers(text_model, inputs_embeds[:, d_idx], position_ids[:, :, d_idx], layers[:L], draft_cache, 0)
        if repair_ratio <= 0 or not q_pos:
            b_idx = torch.tensor(q_pos, device=device) if q_pos else d_idx[:0]
            if q_pos:
                mask = _index_mask(b_idx, torch.cat([d_idx, b_idx]), dtype, device)
                h_b = _run_layers(
                    text_model, inputs_embeds[:, b_idx], position_ids[:, :, b_idx],
                    layers[:L], draft_cache, len(d_pos), mask,
                )
            else:
                h_b = None
            perm = torch.tensor([loc[p] for p in d_pos + q_pos], device=device)
            # draft_cache is [D][added]. Inverse-permute into original keep order.
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel(), device=device)
            for i in range(L):
                early.key_cache.append(draft_cache.key_cache[i].index_select(-2, inv))
                early.value_cache.append(draft_cache.value_cache[i].index_select(-2, inv))
            early._seen_tokens = len(pos_list)
            late_in = inputs_embeds.new_empty((1, len(pos_list), inputs_embeds.shape[-1]))
            late_in.index_copy_(1, torch.tensor([loc[p] for p in d_pos], device=device), h_d)
            if h_b is not None:
                late_in.index_copy_(1, torch.tensor([loc[p] for p in q_pos], device=device), h_b)
        else:
            exact = _empty_cache()
            h0 = _run_layers(text_model, inputs_embeds[:, full_pos], position_ids[:, :, full_pos], layers[:1], exact, 0)
            h0_d = _run_layers(text_model, inputs_embeds[:, d_idx], position_ids[:, :, d_idx], layers[:1], _empty_cache(), 0)
            d_slots = torch.tensor([loc[p] for p in d_pos], device=device)
            err = (h0[0].index_select(0, d_slots) - h0_d[0]).float().norm(dim=-1)
            n_fix = min(len(d_pos), int(round(repair_ratio * len(d_pos))))
            fix = set()
            if n_fix:
                chosen = torch.topk(err, n_fix).indices.tolist()
                fix = {d_pos[i] for i in chosen}
            keep_d = [p for p in d_pos if p not in fix]
            added = [p for p in pos_list if p in fix or p not in draft_set]
            for i in range(L):
                early.key_cache.append(exact.key_cache[0].new_empty(exact.key_cache[0].shape[:2] + (len(pos_list),) + exact.key_cache[0].shape[3:]))
                early.value_cache.append(early.key_cache[-1].new_empty(early.key_cache[-1].shape))
            early.key_cache[0] = exact.key_cache[0]
            early.value_cache[0] = exact.value_cache[0]
            h_add = h0
            if L > 1 and added:
                add_idx = torch.tensor(added, device=device)
                add_slots = torch.tensor([loc[p] for p in added], device=device)
                keep_slots = torch.tensor([loc[p] for p in keep_d], device=device) if keep_d else None
                src_slots = torch.tensor([d_pos.index(p) for p in keep_d], device=device) if keep_d else None
                h_cur = h0.index_select(1, add_slots)
                fix_cache = _empty_cache()
                for li in range(1, L):
                    if keep_d:
                        _seed_layer(fix_cache, li, draft_cache.key_cache[li].index_select(-2, src_slots), draft_cache.value_cache[li].index_select(-2, src_slots))
                    k_idx = torch.tensor(keep_d + added, device=device)
                    mask = _index_mask(add_idx, k_idx, dtype, device)
                    h_cur = _run_layers(
                        text_model, h_cur, position_ids[:, :, add_idx],
                        layers[li:li + 1], fix_cache, len(keep_d), mask,
                    )
                    k = fix_cache.key_cache[li]
                    v = fix_cache.value_cache[li]
                    if keep_d:
                        early.key_cache[li].index_copy_(-2, keep_slots, k[..., :len(keep_d), :])
                        early.value_cache[li].index_copy_(-2, keep_slots, v[..., :len(keep_d), :])
                        early.key_cache[li].index_copy_(-2, add_slots, k[..., len(keep_d):, :])
                        early.value_cache[li].index_copy_(-2, add_slots, v[..., len(keep_d):, :])
                    else:
                        early.key_cache[li].index_copy_(-2, add_slots, k)
                        early.value_cache[li].index_copy_(-2, add_slots, v)
                h_add = h0.clone()
                h_add.index_copy_(1, add_slots, h_cur)
            late_in = inputs_embeds.new_empty((1, len(pos_list), inputs_embeds.shape[-1]))
            late_in.index_copy_(1, d_slots, h_d)
            if added:
                add_slots = torch.tensor([loc[p] for p in added], device=device)
                # Repaired draft tokens and new tokens take the recomputed state; others keep the draft path.
                late_in.index_copy_(1, add_slots, h_add.index_select(1, add_slots))
            early._seen_tokens = len(pos_list)
            LAST["splice_repair"] = int(len(fix) if repair_ratio > 0 else 0)
    late = _empty_cache()
    if L < n_layers:
        hidden = _run_layers(text_model, late_in, position_ids[:, :, full_pos], layers[L:], late, 0)
    else:
        hidden = late_in
    hidden = text_model.norm(hidden)
    joined = _empty_cache()
    joined.key_cache = list(early.key_cache) + list(late.key_cache)
    joined.value_cache = list(early.value_cache) + list(late.value_cache)
    joined._seen_tokens = len(pos_list)
    thinker._splice_state = {
        "cache": joined,
        "length": len(pos_list),
        "next_pos": position_ids[:, :, full_pos][:, :, -1:] + 1,
    }
    LAST["exec"] = "splice"
    LAST["splice_l"] = L
    LAST["splice_r"] = repair_ratio
    LAST["splice_sec"] = round(time.perf_counter() - t0, 3)
    print("splice", "L", L, "r", repair_ratio, "draft", len(d_pos), "full", len(pos_list), "new", len(q_pos), "sec", LAST["splice_sec"], flush=True)
    LAST.pop("draft_pos", None)
    LAST.pop("question_at", None)
    return hidden


def greedy_splice(thinker, first_logits, max_new_tokens, eos_id):
    state = thinker._splice_state
    logits = first_logits[0, -1]
    ids = []
    text_model = thinker.model
    layers = list(text_model.layers)
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        if eos_id is not None and nxt == int(eos_id):
            break
        emb = text_model.embed_tokens(torch.tensor([[nxt]], device=logits.device))
        pos = state["next_pos"]
        hidden = _run_layers(text_model, emb, pos, layers, state["cache"], state["length"])
        state["length"] += 1
        state["next_pos"] = pos + 1
        hidden = text_model.norm(hidden)
        logits = thinker.lm_head(hidden[:, -1])[0]
    return ids


def greedy_reuse(thinker, first_logits, max_new_tokens, eos_id):
    state = thinker._reuse_state
    logits = first_logits[0, -1]
    ids = []
    text_model = thinker.model
    layers = list(text_model.layers)
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        if eos_id is not None and nxt == int(eos_id):
            break
        emb = text_model.embed_tokens(torch.tensor([[nxt]], device=logits.device))
        pos = state["next_pos"]
        hidden = _run_layers(text_model, emb, pos, layers[:4], state["early"], state["early_len"])
        state["early_len"] += 1
        hidden = _run_layers(text_model, hidden, pos, layers[4:], state["late"], state["full_len"])
        state["full_len"] += 1
        state["next_pos"] = pos + 1
        hidden = text_model.norm(hidden)
        logits = thinker.lm_head(hidden[:, -1])[0]
    return ids
