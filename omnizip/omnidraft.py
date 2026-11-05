"""Draft-guided spatiotemporal anchor selection.

The draft is the first part of the spatiotemporal anchor pick order (video)
and a coarser cut of the same anchor merge tree (audio). draft_ratio is a
fraction of all tokens of that modality, then capped by the keep budget. The draft prefix runs
through the first layers once; a real-question suffix and a null-question
suffix read it from the cache. Their attention difference, pooled per time
slice over both modalities, sets how many of the remaining tokens each slice
receives. Inside a slice the spatiotemporal anchor greedy still decides which
tokens. With lam=0 the final set is exactly the spatiotemporal anchor set.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from omnizip.fastsel import allocate, audio_merge_tree, audio_undo, video_graph, video_order, _cover
from omnizip.omnispan import align_mel, video_coords

LAST = {}
SNAP = {}


def _suffix_mask(d, s, dtype, device):
    mask = torch.full((1, 1, s, d + s), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[..., :d] = 0
    mask[0, 0, :, d:] = torch.triu(mask[0, 0, :, d:], diagonal=1)
    return mask


@torch.no_grad()
def _layer_sdpa(layer, hidden, position_embeddings, cache, cache_position, mask, need_weights=False):
    """Explicit mask. Flash's causal mask is top-left and wrong for a suffix on a prefix cache."""
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb, repeat_kv
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attn = layer.self_attn
    bsz, q_len, _ = normed.shape
    query = attn.q_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    value = attn.v_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, attn.rope_scaling["mrope_section"])
    if cache is not None:
        key, value = cache.update(key, value, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
    key = repeat_kv(key, attn.num_key_value_groups)
    value = repeat_kv(value, attn.num_key_value_groups)
    weights = None
    if need_weights:
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * (attn.head_dim ** -0.5)
        if mask is not None:
            logits = logits + mask.float()
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights.to(value.dtype), value)
    else:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=False)
    out = attn.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))
    hidden = residual + out
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, weights


def _snap_cache(cache, n, hidden):
    keys = [cache.key_cache[i].clone() for i in range(n)]
    vals = [cache.value_cache[i].clone() for i in range(n)]
    return hidden.detach(), keys, vals


def probe_attention(thinker, embeds, pos, draft_cols, suffixes, n_layers, snap_layers=None, continue_to=None, detail=None):
    """Mean attention (heads, suffix tokens) from each suffix to each draft column at layer n_layers."""
    from transformers.cache_utils import DynamicCache
    tm = thinker.model
    layers = list(tm.layers[:n_layers])
    d = int(draft_cols.numel())
    cache = DynamicCache()
    h = embeds[:, draft_cols]
    p = pos[:, :, draft_cols]
    pe = tm.rotary_emb(h, p)
    cp = torch.arange(d, device=embeds.device)
    snaps = {}
    want = set(snap_layers or ())
    for i, layer in enumerate(layers):
        h = layer(h, attention_mask=None, position_ids=p, past_key_value=cache,
                  use_cache=True, cache_position=cp, position_embeddings=pe)[0]
        if (i + 1) in want:
            snaps[i + 1] = _snap_cache(cache, i + 1, h)
    out = []
    for hs, ps in suffixes:
        s = int(hs.shape[1])
        mask = _suffix_mask(d, s, hs.dtype, hs.device)
        pe_s = tm.rotary_emb(hs, ps)
        cps = torch.arange(d, d + s, device=hs.device)
        w = None
        for i, layer in enumerate(layers):
            hs, w = _layer_sdpa(layer, hs, pe_s, cache, cps, mask, need_weights=(i == len(layers) - 1))
        out.append(w[0, :, :, :d].float().mean(dim=(0, 1)))
        if detail is not None:
            detail.setdefault("heads", []).append(w[0, :, :, :d].float().mean(dim=1).detach().cpu())
            vv = cache.value_cache[len(layers) - 1].detach().float().norm(dim=-1).mean(dim=1)[0, :d]
            detail["v"] = vv.cpu()
        cache.crop(d)
    if continue_to and continue_to > n_layers:
        for i, layer in enumerate(tm.layers[n_layers:continue_to], start=n_layers):
            h = layer(h, attention_mask=None, position_ids=p, past_key_value=cache,
                      use_cache=True, cache_position=cp, position_embeddings=pe)[0]
            if (i + 1) in want or (i + 1) == continue_to:
                snaps[i + 1] = _snap_cache(cache, i + 1, h)
    return out, snaps


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
    if cfg.get("signal") == "oracle":
        return _oracle_mask(
            thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
            video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
            audio_scores, video_scores, cfg,
        )
    if int(cfg.get("rounds", 1)) == 2:
        return _two_round_mask(
            thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
            video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
            audio_scores, video_scores, cfg,
        )
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
        b_v = int(min(k_v, max(n_frames, round(rho * n_v))))
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
        b_a = int(min(k_a, max(1, round(rho * n_a))))
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
    boost = cfg.get("boost") or ""
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
        snap_layers = None
        cont = None
        if cfg.get("exec") == "splice":
            depth = int(cfg.get("splice_l", 12))
            snap_layers = set()
            if 0 < depth <= n_layers:
                snap_layers.add(depth)
            if float(cfg.get("splice_r", 0.0)) > 0 and depth > 0:
                snap_layers.add(min(1, depth))
            if depth > n_layers:
                cont = depth
        detail = {} if cfg.get("boost") == "headval" else None
        probed, snaps = probe_attention(
            thinker, inputs_embeds, position_ids, draft_cols, suffixes, n_layers, snap_layers, cont, detail,
        )
        if snaps:
            SNAP.clear()
            SNAP.update(snaps)
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
        boost = cfg.get("boost") or ""
        if boost == "headval" and detail and use_null and len(detail.get("heads", [])) >= 2:
            excess = _head_value_excess(detail["heads"][0].numpy(), detail["heads"][1].numpy(), detail["v"].numpy())
        elif boost == "grad":
            excess = _grad_score(thinker, inputs_embeds, position_ids, draft_cols, real, cfg)
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
        if boost == "conf" and lam > 0:
            margin = _draft_margin(thinker, inputs_embeds, position_ids, draft_cols, real, cfg)
            lam = lam * (1.0 - margin)
            LAST["draft_margin"] = round(margin, 4)
        if boost == "gap" and rel_v is not None and order is not None:
            rel_v = _gap_relevance(graph, frames, order[:b_v], rel_v, n_slices)
            rel = rel_v
        if cfg.get("signal") == "shuffle" and rel_v is not None:
            perm = np.random.RandomState(0).permutation(n_slices)
            rel_v = np.asarray(rel_v)[perm]
            if rel_a is not None:
                rel_a = np.asarray(rel_a)[perm]
            rel = rel_v

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
        token_mul = None
        seed = None
        if boost == "neigh" and excess is not None:
            token_mul = _neighbor_mul(graph, v_np, cols, excess, order[:b_v], lam)
            seed = order[:b_v]
        final_v = video_order(graph, vscores, coords, k_v, caps=caps, cap_from=b_v, seed_order=seed, token_mul=token_mul)
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
        "boost": cfg.get("boost") or "",
    })
    print("omnidraft", " ".join(f"{k}={v}" for k, v in LAST.items()), flush=True)
    LAST["draft_pos"] = saved_draft
    LAST["question_at"] = saved_start
    return keep


def _layer_flash_suffix(layer, hidden, position_embeddings, cache, cache_position):
    """New tokens are a suffix of the draft cache. Flash causal is bottom-right, so each of them sees the whole draft plus earlier new tokens."""
    from flash_attn import flash_attn_func
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb, repeat_kv
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attn = layer.self_attn
    bsz, q_len, _ = normed.shape
    query = attn.q_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    value = attn.v_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, attn.rope_scaling["mrope_section"])
    key, value = cache.update(key, value, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
    key = repeat_kv(key, attn.num_key_value_groups)
    value = repeat_kv(value, attn.num_key_value_groups)
    # flash_attn_func wants (batch, seq, heads, dim). causal=True aligns to the bottom right.
    out = flash_attn_func(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        dropout_p=0.0, causal=True,
    )
    out = attn.o_proj(out.reshape(bsz, q_len, -1))
    hidden = residual + out
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden


def _run_layers(text_model, hidden, position_ids, layers, cache, past_len, mask=None, causal_suffix=False):
    seq = hidden.shape[1]
    cache_position = torch.arange(past_len, past_len + seq, device=hidden.device)
    position_embeddings = text_model.rotary_emb(hidden, position_ids)
    # Real mRoPE is already in position_embeddings. Non-monotonic ids make Flash take the packed varlen path.
    step_ids = position_ids
    if causal_suffix:
        step_ids = torch.arange(seq, device=hidden.device).view(1, 1, -1).expand(3, 1, -1)
    for layer in layers:
        if causal_suffix or mask is None:
            hidden = layer(
                hidden, attention_mask=None, position_ids=step_ids, past_key_value=cache,
                output_attentions=False, use_cache=True, cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        else:
            hidden = _layer_sdpa(layer, hidden, position_embeddings, cache, cache_position, mask)[0]
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


class _SlotCache:
    """KV buffer with spare room. Decode writes the new token in place instead of concatenating the whole cache."""

    def __init__(self, keys, values, length, extra):
        self.length = int(length)
        self.keys = []
        self.values = []
        pairs = [(k, v) for k, v in zip(keys, values) if k.ndim == 4]
        for key, value in pairs:
            shape = key.shape[:2] + (self.length + extra,) + key.shape[3:]
            buf_k = key.new_empty(shape)
            buf_v = value.new_empty(shape)
            buf_k[:, :, : self.length] = key
            buf_v[:, :, : self.length] = value
            self.keys.append(buf_k)
            self.values.append(buf_v)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        q = key.shape[-2]
        if layer_idx == 0:
            start = self.length
            self.length += q
        else:
            start = self.length - q
        self.keys[layer_idx][:, :, start : start + q] = key
        self.values[layer_idx][:, :, start : start + q] = value
        end = start + q
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]


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
        saved = SNAP.get(L)
        if saved is None:
            raise RuntimeError(f"splice is missing the layer-{L} draft checkpoint")
        h_d, keys, vals = saved
        draft_cache = _empty_cache()
        for i in range(L):
            draft_cache.key_cache.append(keys[i])
            draft_cache.value_cache.append(vals[i])
        draft_cache._seen_tokens = len(d_pos)
        if repair_ratio <= 0 or not q_pos:
            b_idx = torch.tensor(q_pos, device=device) if q_pos else d_idx[:0]
            if q_pos:
                if cfg.get("vis") == "index":
                    # Original-order causal: a new token sees a draft token only when that draft token is earlier.
                    k_idx = torch.tensor(d_pos + q_pos, device=device)
                    mask = _index_mask(b_idx, k_idx, inputs_embeds.dtype, device)
                    h_b = _run_layers(
                        text_model, inputs_embeds[:, b_idx], position_ids[:, :, b_idx],
                        layers[:L], draft_cache, len(d_pos), mask=mask,
                    )
                else:
                    # Appended after the draft. Each new token sees the draft cache and earlier new tokens.
                    h_b = _run_layers(
                        text_model, inputs_embeds[:, b_idx], position_ids[:, :, b_idx],
                        layers[:L], draft_cache, len(d_pos), causal_suffix=True,
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
            h0_d = SNAP[1][0]
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
                if keep_d:
                    for li in range(1, L):
                        _seed_layer(
                            fix_cache, li,
                            draft_cache.key_cache[li].index_select(-2, src_slots),
                            draft_cache.value_cache[li].index_select(-2, src_slots),
                        )
                # Repaired draft tokens and new tokens are one Flash suffix of the unrepaired draft.
                h_cur = _run_layers(
                    text_model, h_cur, position_ids[:, :, add_idx],
                    layers[1:L], fix_cache, len(keep_d), causal_suffix=True,
                )
                for li in range(1, L):
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
    joined.key_cache = [t.contiguous() for t in list(early.key_cache) + list(late.key_cache)]
    joined.value_cache = [t.contiguous() for t in list(early.value_cache) + list(late.value_cache)]
    joined._seen_tokens = len(pos_list)
    thinker._splice_state = {
        "cache": _SlotCache(joined.key_cache, joined.value_cache, len(pos_list), 16),
        # layers past L are stored at their real index, with empty placeholders in front
        "length": len(pos_list),
        "next_pos": position_ids[:, :, full_pos][:, :, -1:] + 1,
    }
    LAST["exec"] = "splice"
    LAST["splice_l"] = L
    LAST["splice_r"] = repair_ratio
    if hidden.is_cuda:
        torch.cuda.synchronize()
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
    t_decode = time.perf_counter()
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
    if logits.is_cuda:
        torch.cuda.synchronize()
    print("decode_sec", round(time.perf_counter() - t_decode, 3), flush=True)
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


def _media_plan(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores, video_scores, cfg,
):
    """Anchor order plus the draft cut. Shared by the two-round selector and the fused run."""
    rho = float(cfg.get("draft_ratio", 0.5))
    ids = input_ids[0]
    emb = inputs_embeds[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    n_a, n_v = int(a_pos.numel()), int(v_pos.numel())
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
        b_v = int(min(k_v, max(n_frames, round(rho * n_v))))
    tree = None
    a_slice = None
    k_a = b_a = 0
    h = c = None
    if n_a:
        h = emb[a_pos].detach()
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, n_a)
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        b_a = int(min(k_a, max(1, round(rho * n_a))))
        tree = audio_merge_tree(h, audio_scores, c, b_a, k_a)
        if frames is not None:
            vt = tpos[v_pos.cpu().numpy()]
            starts = np.array([vt[np.argmax(frames == k)] for k in range(n_slices)])
            a_slice = np.clip(np.searchsorted(starts, tpos[a_pos.cpu().numpy()], side="right") - 1, 0, n_slices - 1)
        else:
            a_slice = np.zeros(n_a, dtype=np.int64)
    a_np = a_pos.cpu().numpy()
    v_np = v_pos.cpu().numpy()
    start = int(max(a_np.max() if n_a else 0, v_np.max() if n_v else 0)) + 1
    return {
        "ids": ids, "device": device, "a_pos": a_pos, "v_pos": v_pos, "a_np": a_np, "v_np": v_np,
        "n_a": n_a, "n_v": n_v, "coords": coords, "frames": frames, "order": order, "graph": graph,
        "vscores": vscores, "n_slices": n_slices, "k_v": k_v, "b_v": b_v, "tree": tree,
        "a_slice": a_slice, "k_a": k_a, "b_a": b_a, "start": start,
    }


def _cols(n, a_np, v_np, v_idx, a_idx, start, device):
    media = np.zeros(n, dtype=bool)
    if len(a_np):
        media[a_np] = True
    if len(v_np):
        media[v_np] = True
    draft = ~media
    draft[start:] = False
    if len(v_idx):
        draft[v_np[np.asarray(v_idx, dtype=np.int64)]] = True
    if a_idx:
        draft[a_np[np.asarray(list(a_idx), dtype=np.int64)]] = True
    return torch.from_numpy(np.flatnonzero(draft)).to(device)


def _relevance(excess, cols, a_np, v_np, a_slice, frames, n_slices, n):
    kind = np.full(n, -1, dtype=np.int64)
    slc = np.zeros(n, dtype=np.int64)
    if len(a_np):
        kind[a_np] = 0
        slc[a_np] = a_slice
    if len(v_np):
        kind[v_np] = 1
        slc[v_np] = frames
    on = kind[cols] >= 0
    if not on.any():
        return None
    return _slice_relevance(excess[on], kind[cols][on], slc[cols][on], n_slices)


def _probe_rel(thinker, embeds, pos, ids, cols, start, n_layers, cfg, plan):
    real = (embeds[:, start:], pos[:, :, start:])
    suffixes = [real, _null_suffix(thinker, ids, pos, start, cfg)]
    probed, _ = probe_attention(thinker, embeds, pos, cols, suffixes, n_layers)
    excess = torch.clamp(probed[0] - probed[1], min=0).cpu().numpy()
    rel = _relevance(excess, cols.cpu().numpy(), plan["a_np"], plan["v_np"], plan["a_slice"], plan["frames"], plan["n_slices"], int(ids.shape[0]))
    lam = float(cfg.get("lam", 0.0)) if rel is not None else 0.0
    return rel, lam


def _round_caps(slice_of, pool, add_n, rel, lam, n_slices, taken, capacity):
    pool = np.asarray(pool, dtype=np.int64)
    add_n = int(max(0, min(int(add_n), int(pool.shape[0]))))
    default = np.bincount(slice_of[pool[:add_n]], minlength=n_slices).astype(np.float64) if add_n else np.zeros(n_slices)
    dist = default / max(1, add_n)
    if lam > 0 and rel is not None:
        dist = (1.0 - lam) * dist + lam * np.asarray(rel, dtype=np.float64)
    room = np.maximum(np.asarray(capacity) - np.asarray(taken), 0)
    return allocate(add_n, dist, room)


def _two_steps(plan, rel1, lam1, rel2, lam2):
    """Return (video local indices, audio local indices) after draft, round 1, round 2."""
    v_idx = np.zeros(0, dtype=np.int64)
    a_idx = []
    order = plan["order"]
    if order is not None:
        frames = plan["frames"]
        k_v, b_v, n_slices = plan["k_v"], plan["b_v"], plan["n_slices"]
        m1 = (k_v - b_v) // 2
        cap = np.bincount(frames, minlength=n_slices)
        taken = np.bincount(frames[order[:b_v]], minlength=n_slices)
        caps = _round_caps(frames, order[b_v:], m1, rel1, lam1, n_slices, taken, cap)
        order1 = video_order(plan["graph"], plan["vscores"], plan["coords"], b_v + m1, caps=caps, cap_from=b_v)
        rest = np.asarray([i for i in order.tolist() if i not in set(order1.tolist())], dtype=np.int64)
        m2 = k_v - int(order1.shape[0])
        taken = np.bincount(frames[order1], minlength=n_slices)
        caps = _round_caps(frames, rest, m2, rel2, lam2, n_slices, taken, cap)
        v_idx = video_order(plan["graph"], plan["vscores"], plan["coords"], k_v, caps=caps, seed_order=order1)
    tree = plan["tree"]
    if tree is not None:
        a_slice = plan["a_slice"]
        k_a, b_a, n_slices = plan["k_a"], plan["b_a"], plan["n_slices"]
        m1 = (k_a - b_a) // 2
        cap = np.bincount(a_slice, minlength=n_slices)
        base = list(tree["final"])
        taken = np.bincount(a_slice[base], minlength=n_slices)
        added = np.asarray(sorted(set(tree["snap"]) - set(tree["final"])), dtype=np.int64)
        caps = _round_caps(a_slice, added, m1, rel1, lam1, n_slices, taken, cap)
        audio_undo(tree, b_a + m1, a_slice, caps)
        replay = list(tree.get("undo_trace") or [])
        anchors1 = set(tree["final"])
        # undo_trace is the round-1 undos; anchors are not returned. Rebuild from replay.
        anchors1 = set(audio_undo(tree, b_a + m1, a_slice, caps))
        replay = list(tree.get("undo_trace") or [])
        rest = np.asarray(sorted(set(tree["snap"]) - anchors1), dtype=np.int64)
        m2 = k_a - len(anchors1)
        taken = np.bincount(a_slice[sorted(anchors1)], minlength=n_slices)
        caps = _round_caps(a_slice, rest, m2, rel2, lam2, n_slices, taken, cap)
        a_idx = audio_undo(tree, k_a, a_slice, caps, replay=replay)
    return v_idx, a_idx


def _one_step(plan, rel, lam, which):
    """which=1 returns the set after the first completion; which=2 is unused."""
    v_idx = np.zeros(0, dtype=np.int64)
    a_idx = []
    order = plan["order"]
    if order is not None:
        frames = plan["frames"]
        k_v, b_v, n_slices = plan["k_v"], plan["b_v"], plan["n_slices"]
        m1 = (k_v - b_v) // 2
        cap = np.bincount(frames, minlength=n_slices)
        taken = np.bincount(frames[order[:b_v]], minlength=n_slices)
        caps = _round_caps(frames, order[b_v:], m1, rel, lam, n_slices, taken, cap)
        v_idx = video_order(plan["graph"], plan["vscores"], plan["coords"], b_v + m1, caps=caps, cap_from=b_v)
    tree = plan["tree"]
    if tree is not None:
        a_slice = plan["a_slice"]
        k_a, b_a, n_slices = plan["k_a"], plan["b_a"], plan["n_slices"]
        m1 = (k_a - b_a) // 2
        cap = np.bincount(a_slice, minlength=n_slices)
        base = list(tree["final"])
        taken = np.bincount(a_slice[base], minlength=n_slices)
        added = np.asarray(sorted(set(tree["snap"]) - set(base)), dtype=np.int64)
        caps = _round_caps(a_slice, added, m1, rel, lam, n_slices, taken, cap)
        a_idx = audio_undo(tree, b_a + m1, a_slice, caps)
    return v_idx, a_idx


def _finish_step(plan, v_so_far, a_so_far, rel, lam):
    v_idx = v_so_far
    a_idx = list(a_so_far)
    order = plan["order"]
    if order is not None:
        frames = plan["frames"]
        k_v, n_slices = plan["k_v"], plan["n_slices"]
        have = set(np.asarray(v_so_far).tolist())
        rest = np.asarray([i for i in order.tolist() if i not in have], dtype=np.int64)
        m2 = k_v - len(have)
        cap = np.bincount(frames, minlength=n_slices)
        taken = np.bincount(frames[np.asarray(sorted(have), dtype=np.int64)], minlength=n_slices) if have else np.zeros(n_slices, dtype=np.int64)
        caps = _round_caps(frames, rest, m2, rel, lam, n_slices, taken, cap)
        v_idx = video_order(plan["graph"], plan["vscores"], plan["coords"], k_v, caps=caps, seed_order=np.asarray(v_so_far, dtype=np.int64))
    tree = plan["tree"]
    if tree is not None:
        a_slice = plan["a_slice"]
        k_a, n_slices = plan["k_a"], plan["n_slices"]
        anchors = set(a_so_far)
        rest = np.asarray(sorted(set(tree["snap"]) - anchors), dtype=np.int64)
        m2 = k_a - len(anchors)
        cap = np.bincount(a_slice, minlength=n_slices)
        taken = np.bincount(a_slice[sorted(anchors)], minlength=n_slices) if anchors else np.zeros(n_slices, dtype=np.int64)
        caps = _round_caps(a_slice, rest, m2, rel, lam, n_slices, taken, cap)
        replay = list(tree.get("undo_trace") or [])
        a_idx = audio_undo(tree, k_a, a_slice, caps, replay=replay)
    return v_idx, a_idx


def _keep_from(plan, v_idx, a_idx):
    ids = plan["ids"]
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=plan["device"])
    if plan["n_a"]:
        drop = torch.ones(plan["n_a"], dtype=torch.bool)
        if a_idx:
            drop[torch.as_tensor(sorted(a_idx), dtype=torch.long)] = False
        keep[plan["a_pos"][drop.to(plan["device"])]] = False
    if plan["n_v"] and plan["order"] is not None:
        drop = torch.ones(plan["n_v"], dtype=torch.bool)
        drop[torch.as_tensor(np.sort(v_idx), dtype=torch.long)] = False
        keep[plan["v_pos"][drop.to(plan["device"])]] = False
    return keep


def _two_round_mask(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores, video_scores, cfg,
):
    """Two exact probes (layer round_l1, then probe_layer). The answer is a full rerun of the final set."""
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    plan = _media_plan(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
        audio_scores, video_scores, cfg,
    )
    ids = plan["ids"]
    l1 = int(cfg.get("round_l1", 4))
    l2 = int(cfg.get("probe_layer", 12))
    v0 = plan["order"][: plan["b_v"]] if plan["order"] is not None else []
    a0 = list(plan["tree"]["final"]) if plan["tree"] is not None else []
    cols = _cols(int(ids.shape[0]), plan["a_np"], plan["v_np"], v0, a0, plan["start"], plan["device"])
    rel1, lam1 = _probe_rel(thinker, inputs_embeds, position_ids, ids, cols, plan["start"], l1, cfg, plan)
    v1, a1 = _one_step(plan, rel1, lam1, 1)
    cols = _cols(int(ids.shape[0]), plan["a_np"], plan["v_np"], v1, a1, plan["start"], plan["device"])
    rel2, lam2 = _probe_rel(thinker, inputs_embeds, position_ids, ids, cols, plan["start"], l2, cfg, plan)
    v2, a2 = _finish_step(plan, v1, a1, rel2, lam2)
    LAST.clear()
    LAST.update({"exec": "twoselect", "k_video": int(plan["k_v"]), "k_audio": int(plan["k_a"]), "lam": lam2})
    return _keep_from(plan, v2, a2)


def _reorder(cache, n_layers, src_pos, hidden):
    order = sorted(range(len(src_pos)), key=lambda i: src_pos[i])
    idx = torch.tensor(order, device=hidden.device)
    for i in range(n_layers):
        cache.key_cache[i] = cache.key_cache[i].index_select(-2, idx).contiguous()
        cache.value_cache[i] = cache.value_cache[i].index_select(-2, idx).contiguous()
    cache._seen_tokens = len(src_pos)
    return hidden.index_select(1, idx), [src_pos[i] for i in order]


def _suffix_on(thinker, embeds, pos, ids, cache, prefix_pos, depth, start, cfg):
    """Attention of the real and null questions onto prefix_pos, using the cache already at `depth`."""
    d = len(prefix_pos)
    real = (embeds[:, start:], pos[:, :, start:])
    suffixes = [real, _null_suffix(thinker, ids, pos, start, cfg)]
    tm = thinker.model
    layers = list(tm.layers[:depth])
    out = []
    for hs, ps in suffixes:
        s = int(hs.shape[1])
        mask = _suffix_mask(d, s, hs.dtype, hs.device)
        pe = tm.rotary_emb(hs, ps)
        cp = torch.arange(d, d + s, device=hs.device)
        w = None
        for i, layer in enumerate(layers):
            hs, w = _layer_sdpa(layer, hs, pe, cache, cp, mask, need_weights=(i == len(layers) - 1))
        out.append(w[0, :, :, :d].float().mean(dim=(0, 1)))
        cache.crop(d)
    return out


def prefill_two(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_scores, video_scores, keep_ratio, cfg,
):
    """Two-round selection whose second probe reads the fused layer-12 cache. Each media token is computed once."""
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    plan = _media_plan(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, thinker.config.audio_token_id, thinker.config.video_token_id,
        keep_ratio, int(thinker.visual.spatial_merge_size), audio_scores, video_scores, cfg,
    )
    ids = plan["ids"]
    device = plan["device"]
    l1 = int(cfg.get("round_l1", 4))
    l2 = int(cfg.get("probe_layer", 12))
    text_model = thinker.model
    layers = list(text_model.layers)
    v0 = plan["order"][: plan["b_v"]] if plan["order"] is not None else []
    a0 = list(plan["tree"]["final"]) if plan["tree"] is not None else []
    d_cols = _cols(int(ids.shape[0]), plan["a_np"], plan["v_np"], v0, a0, plan["start"], device)
    d_pos = [int(x) for x in d_cols.tolist()]
    cache = _empty_cache()
    h = _run_layers(text_model, inputs_embeds[:, d_cols], position_ids[:, :, d_cols], layers[:l1], cache, 0)
    probed = _suffix_on(thinker, inputs_embeds, position_ids, ids, cache, d_pos, l1, plan["start"], cfg)
    excess = torch.clamp(probed[0] - probed[1], min=0).cpu().numpy()
    rel1 = _relevance(excess, np.asarray(d_pos), plan["a_np"], plan["v_np"], plan["a_slice"], plan["frames"], plan["n_slices"], int(ids.shape[0]))
    lam1 = float(cfg.get("lam", 0.0)) if rel1 is not None else 0.0
    v1, a1 = _one_step(plan, rel1, lam1, 1)
    u_cols = _cols(int(ids.shape[0]), plan["a_np"], plan["v_np"], v1, a1, plan["start"], device)
    u_pos = [p for p in u_cols.tolist() if p not in set(d_pos)]
    if u_pos:
        u_idx = torch.tensor(u_pos, device=device)
        k_idx = torch.tensor(d_pos + u_pos, device=device)
        mask = _index_mask(u_idx, k_idx, inputs_embeds.dtype, device)
        h_u = _run_layers(text_model, inputs_embeds[:, u_idx], position_ids[:, :, u_idx], layers[:l1], cache, len(d_pos), mask=mask)
        h = torch.cat([h, h_u], dim=1)
        src = d_pos + u_pos
    else:
        src = d_pos
    h, src = _reorder(cache, l1, src, h)
    h = _run_layers(text_model, h, position_ids[:, :, torch.tensor(src, device=device)], layers[l1:l2], cache, 0)
    probed = _suffix_on(thinker, inputs_embeds, position_ids, ids, cache, src, l2, plan["start"], cfg)
    excess = torch.clamp(probed[0] - probed[1], min=0).cpu().numpy()
    rel2 = _relevance(excess, np.asarray(src), plan["a_np"], plan["v_np"], plan["a_slice"], plan["frames"], plan["n_slices"], int(ids.shape[0]))
    lam2 = float(cfg.get("lam", 0.0)) if rel2 is not None else 0.0
    v2, a2 = _finish_step(plan, v1, a1, rel2, lam2)
    keep = _keep_from(plan, v2, a2)
    full = [int(p) for p in torch.nonzero(keep, as_tuple=True)[0].tolist()]
    have = set(src)
    new_pos = [p for p in full if p not in have]
    if new_pos:
        n_idx = torch.tensor(new_pos, device=device)
        k_idx = torch.tensor(src + new_pos, device=device)
        mask = _index_mask(n_idx, k_idx, inputs_embeds.dtype, device)
        h_n = _run_layers(text_model, inputs_embeds[:, n_idx], position_ids[:, :, n_idx], layers[:l2], cache, len(src), mask=mask)
        h = torch.cat([h, h_n], dim=1)
        src = src + new_pos
    h, src = _reorder(cache, l2, src, h)
    full_idx = torch.tensor(src, device=device)
    if l2 < len(layers):
        h = _run_layers(text_model, h, position_ids[:, :, full_idx], layers[l2:], cache, 0)
    h = text_model.norm(h)
    keys = [cache.key_cache[i] for i in range(len(layers)) if i < len(cache.key_cache) and torch.is_tensor(cache.key_cache[i]) and cache.key_cache[i].ndim == 4]
    vals = [cache.value_cache[i] for i in range(len(layers)) if i < len(cache.value_cache) and torch.is_tensor(cache.value_cache[i]) and cache.value_cache[i].ndim == 4]
    thinker._splice_state = {
        "cache": _SlotCache(keys, vals, len(src), 16),
        "length": len(src),
        "next_pos": position_ids[:, :, full_idx][:, :, -1:] + 1,
    }
    LAST.clear()
    LAST.update({"exec": "twofuse", "k_video": int(v2.shape[0]) if hasattr(v2, "shape") else 0, "k_audio": len(a2), "lam": lam2})
    return h


def _option_ids(cfg):
    ids = [int(x) for x in cfg.get("option_token_ids", []) if int(x) >= 0]
    return ids[:4]


def _head_value_excess(wq, w0, vnorm):
    """Head weight is how much that head's map changes between the real and null question."""
    eps = 1e-8
    kl = []
    for h in range(wq.shape[0]):
        p = np.clip(wq[h], 0, None) + eps
        q = np.clip(w0[h], 0, None) + eps
        p = p / p.sum()
        q = q / q.sum()
        kl.append(float(np.sum(p * np.log(p / q))))
    kl = np.asarray(kl, dtype=np.float64)
    kl = kl / max(float(kl.sum()), 1e-12)
    excess = np.maximum(0.0, (wq - w0).T @ kl)
    return excess * np.maximum(np.asarray(vnorm, dtype=np.float64), 0.0)


def _run_draft_suffix(thinker, embeds, pos, draft_cols, suffix, hold_layer=None):
    tm = thinker.model
    hs, ps = suffix
    h = torch.cat([embeds[:, draft_cols], hs], dim=1).detach()
    if hold_layer is not None:
        h = h.requires_grad_(True)
    p = torch.cat([pos[:, :, draft_cols], ps], dim=2)
    n = int(h.shape[1])
    mask = torch.triu(torch.full((n, n), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device), diagonal=1)[None, None]
    pe = tm.rotary_emb(h, p)
    cp = torch.arange(n, device=h.device)
    held = None
    # Flash's packed path asserts on these position ids once autograd is on. SDPA keeps the gradient.
    for i, layer in enumerate(tm.layers):
        h, _ = _layer_sdpa(layer, h, pe, None, cp, mask, need_weights=False)
        if hold_layer is not None and i + 1 == hold_layer:
            h.retain_grad()
            held = h
    logits = thinker.lm_head(tm.norm(h[:, -1:]))[0, -1]
    return logits, held


def _grad_score(thinker, embeds, pos, draft_cols, suffix, cfg):
    opt = _option_ids(cfg)
    d = int(draft_cols.numel())
    if len(opt) < 2:
        return np.zeros(d, dtype=np.float64)
    with torch.enable_grad():
        logits, held = _run_draft_suffix(thinker, embeds, pos, draft_cols, suffix, hold_layer=int(cfg.get("probe_layer", 12)))
    chosen = logits[opt]
    top = torch.topk(chosen, 2)
    margin = top.values[0] - top.values[1]
    margin.backward()
    g = held.grad[0, :d].float().norm(dim=-1).detach().cpu().numpy()
    thinker.zero_grad(set_to_none=True)
    return g


def _draft_margin(thinker, embeds, pos, draft_cols, suffix, cfg):
    opt = _option_ids(cfg)
    if len(opt) < 2:
        return 0.0
    with torch.no_grad():
        logits, _ = _run_draft_suffix(thinker, embeds, pos, draft_cols, suffix, hold_layer=None)
    prob = torch.softmax(logits[opt].float(), dim=0)
    top = torch.topk(prob, 2).values
    return float((top[0] - top[1]).item())


def _gap_relevance(graph, frames, draft_idx, rel, n_slices):
    indptr, indices, data = graph
    n = int(indptr.shape[0] - 1)
    covered = np.zeros(n, dtype=np.float64)
    taken = np.zeros(n, dtype=bool)
    for p in np.asarray(draft_idx).tolist():
        _cover(indptr, indices, data, covered, int(p))
        taken[int(p)] = True
    gap = np.zeros(n_slices, dtype=np.float64)
    cnt = np.zeros(n_slices, dtype=np.float64)
    for i in range(n):
        if taken[i]:
            continue
        gap[frames[i]] += 1.0 - float(covered[i])
        cnt[frames[i]] += 1.0
    gap = np.divide(gap, np.maximum(cnt, 1.0))
    out = np.asarray(rel, dtype=np.float64) * gap
    z = float(out.sum())
    return out / z if z > 1e-12 else np.asarray(rel, dtype=np.float64)


def _neighbor_mul(graph, v_np, cols, excess, draft_idx, lam):
    indptr, indices, data = graph
    n = int(indptr.shape[0] - 1)
    ex = np.zeros(int(v_np.max()) + 1 if len(v_np) else 1, dtype=np.float64)
    # excess is aligned with absolute positions in `cols`
    pos_ex = {int(c): float(e) for c, e in zip(np.asarray(cols).tolist(), np.asarray(excess).tolist())}
    e_local = np.zeros(n, dtype=np.float64)
    draft = np.zeros(n, dtype=bool)
    draft[np.asarray(draft_idx, dtype=np.int64)] = True
    for local, abs_i in enumerate(np.asarray(v_np).tolist()):
        e_local[local] = pos_ex.get(int(abs_i), 0.0)
    r = np.zeros(n, dtype=np.float64)
    r[draft] = e_local[draft]
    for i in np.flatnonzero(~draft):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            if draft[j]:
                s += float(data[p]) * e_local[j]
        r[i] = s
    scale = r / max(float(r.mean()), 1e-12)
    return (1.0 - lam) + lam * scale


def _oracle_mask(
    thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
    video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores, video_scores, cfg,
):
    """Upper bound: rank every media token by the layer-12 excess of the full prefix, then keep the budget."""
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    plan = _media_plan(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
        audio_scores, video_scores, cfg,
    )
    start = plan["start"]
    cols = torch.arange(start, device=plan["device"])
    real = (inputs_embeds[:, start:], position_ids[:, :, start:])
    suffixes = [real, _null_suffix(thinker, plan["ids"], position_ids, start, cfg)]
    probed, _ = probe_attention(thinker, inputs_embeds, position_ids, cols, suffixes, int(cfg.get("probe_layer", 12)))
    excess = torch.clamp(probed[0] - probed[1], min=0).cpu().numpy()
    ex = np.zeros(int(plan["ids"].shape[0]), dtype=np.float64)
    ex[:start] = excess
    ev = ex[plan["v_np"]] if plan["n_v"] else np.zeros(0)
    ea = ex[plan["a_np"]] if plan["n_a"] else np.zeros(0)
    v_idx = np.argsort(-ev)[: plan["k_v"]] if plan["n_v"] else np.zeros(0, dtype=np.int64)
    a_idx = np.argsort(-ea)[: plan["k_a"]].tolist() if plan["n_a"] else []

    def _mass(local, e):
        if len(e) == 0:
            return 0.0
        return float(np.clip(e[np.asarray(local, dtype=np.int64)], 0, None).sum())

    tot_v = _mass(np.arange(len(ev)), ev) + 1e-12
    tot_a = _mass(np.arange(len(ea)), ea) + 1e-12
    anchor_v = plan["order"][: plan["k_v"]] if plan["order"] is not None else []
    cfg_ref = dict(cfg)
    cfg_ref["signal"] = "quota"
    cfg_ref["boost"] = ""
    ref = omnidraft_keep_mask(
        thinker, input_ids, inputs_embeds, position_ids, input_features, feature_attention_mask,
        video_grid_thw, audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
        audio_scores, video_scores, cfg_ref,
    )
    ref_v = np.flatnonzero(ref[plan["v_pos"]].detach().cpu().numpy()) if plan["n_v"] else []
    ref_a = np.flatnonzero(ref[plan["a_pos"]].detach().cpu().numpy()) if plan["n_a"] else []
    LAST["recall_anchor_v"] = round(_mass(anchor_v, ev) / tot_v, 4)
    LAST["recall_omni_v"] = round(_mass(ref_v, ev) / tot_v, 4)
    LAST["recall_oracle_v"] = round(_mass(v_idx, ev) / tot_v, 4)
    LAST["recall_anchor_a"] = round(_mass(list(plan["tree"]["snap"]) if plan["tree"] is not None else [], ea) / tot_a, 4)
    LAST["recall_omni_a"] = round(_mass(ref_a, ea) / tot_a, 4)
    LAST["recall_oracle_a"] = round(_mass(a_idx, ea) / tot_a, 4)
    print("oracle", " ".join(f"{k}={LAST[k]}" for k in list(LAST) if k.startswith("recall_")), flush=True)
    return _keep_from(plan, v_idx, a_idx)
