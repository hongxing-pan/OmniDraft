"""Correct the running audio-anchor and video facility selectors with one draft query.

The draft is the existing anchor-only audio selection plus spatiotemporal video
selection at the same budget. A 4-block Thinker prefix on that draft supplies
query rows. Those rows reweight the original objectives. Forcing every rho to 1
must reproduce the draft selection.
"""
import time

import numpy as np

LAST = {}


def _attn(scores, n):
    if scores is None or len(scores) != n:
        return np.full(n, 1.0 / n)
    attn = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0)
    total = float(attn.sum())
    return attn / total if total > 0 else np.full(n, 1.0 / n)


def _base_weight(attn):
    return 1.0 + len(attn) * attn


def _match_total(base, corrected):
    total = float(np.sum(base))
    got = float(np.sum(corrected))
    if got <= 0 or total <= 0:
        return base.copy()
    return corrected * (total / got)


def video_owner(packed, chosen):
    n = len(packed)
    best_s = np.zeros(n, dtype=np.float64)
    best_j = np.full(n, -1, dtype=np.int64)
    for j in chosen:
        nbrs, sim = packed[int(j)]
        for t, i in enumerate(nbrs):
            s = float(sim[t])
            if s <= 0:
                continue
            i = int(i)
            if s > best_s[i] or (s == best_s[i] and (best_j[i] < 0 or int(j) < best_j[i])):
                best_s[i] = s
                best_j[i] = int(j)
    return best_j


def _lift(rho_kept, kept_sorted, n, owner):
    raw = np.ones(n, dtype=np.float64)
    lookup = {int(j): float(rho_kept[t]) for t, j in enumerate(kept_sorted)}
    for i, j in enumerate(owner):
        if int(j) >= 0 and int(j) in lookup:
            raw[i] = lookup[int(j)]
    mean = float(raw.mean()) if n else 1.0
    if mean <= 0:
        return np.ones(n, dtype=np.float64)
    return raw / mean


def prepare_draft(emb, ids, audio_id, video_id, keep_ratio, video_grid_thw, spatial_merge_size,
                  audio_scores, video_scores, input_features, feature_attention_mask):
    import torch
    from omnizip.omnianchor import compress_tokens
    from omnizip.omnifac import _cosine_edges, _covers, select_video
    from omnizip.omnispan import align_mel, video_coords

    device = emb.device
    a_pos = (ids == audio_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_id).nonzero(as_tuple=True)[0]
    n_a = int(a_pos.numel())
    n_v = int(v_pos.numel())
    bundle = {
        "a_pos": a_pos, "v_pos": v_pos, "n_a": n_a, "n_v": n_v,
        "k_a": n_a, "k_v": n_v, "records": [], "chosen": [],
        "packed": None, "attn_a": None, "attn_v": None,
        "w0_a": None, "w0_v": None, "raw_a": None, "raw_v": None,
        "acoustic": None, "coords": None, "clamped": False,
    }
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    if n_a:
        h = emb[a_pos].detach()
        c = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            c = align_mel(mel, h.shape[0])
        scores = None if audio_scores is None else audio_scores.detach().float().cpu().numpy()
        attn = _attn(scores, n_a)
        k_a = int(max(1, min(n_a, round(keep_ratio * n_a))))
        records, k_a = compress_tokens(h, scores, k_a, "audio", acoustic=c, anchor_only=True)
        chosen = [int(rec["anchor"]) for rec in records]
        drop = torch.ones(n_a, dtype=torch.bool)
        drop[chosen] = False
        keep[a_pos[drop.to(device)]] = False
        bundle.update(records=records, k_a=k_a, attn_a=attn, w0_a=_base_weight(attn), raw_a=h, acoustic=c)
    if n_v and video_grid_thw is not None:
        raw = emb[v_pos].detach().float().cpu().numpy()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(raw.shape[0], tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        k_v = int(max(1, min(n_v, round(keep_ratio * n_v))))
        clamped = k_v < n_frames
        if clamped:
            k_v = n_frames
        scores = None if video_scores is None else video_scores.detach().float().cpu().numpy()
        attn = _attn(scores, n_v)
        packed = _cosine_edges(raw, _covers(coords, True))
        chosen = select_video(raw, scores, coords, k_v, cross=True, use_attn=True, packed=packed)
        drop = torch.ones(n_v, dtype=torch.bool)
        drop[chosen] = False
        keep[v_pos[drop.to(device)]] = False
        bundle.update(
            chosen=chosen, k_v=len(chosen), packed=packed, attn_v=attn, w0_v=_base_weight(attn),
            raw_v=raw, coords=coords, clamped=clamped,
        )
    bundle["keep0"] = keep
    return bundle


def reselect(bundle, rho_audio, rho_video, mode):
    import torch
    from omnizip.omnianchor import compress_tokens
    from omnizip.omnifac import select_video

    mode = mode or "both"
    n_a = bundle["n_a"]
    n_v = bundle["n_v"]
    if mode == "ident" or rho_audio is None:
        rho_a = np.ones(n_a, dtype=np.float64) if n_a else np.ones(0)
    else:
        rho_a = np.asarray(rho_audio, dtype=np.float64)
    if mode == "ident" or rho_video is None:
        rho_v = np.ones(n_v, dtype=np.float64) if n_v else np.ones(0)
    else:
        rho_v = np.asarray(rho_video, dtype=np.float64)
    if mode == "audio":
        rho_v = np.ones(n_v, dtype=np.float64) if n_v else rho_v
    elif mode == "video":
        rho_a = np.ones(n_a, dtype=np.float64) if n_a else rho_a

    device = bundle["keep0"].device
    keep = torch.ones(bundle["keep0"].shape[0], dtype=torch.bool, device=device)
    anchors = []
    if n_a and bundle["raw_a"] is not None:
        kept = sorted(int(rec["anchor"]) for rec in bundle["records"])
        owner = np.full(n_a, -1, dtype=np.int64)
        for rec in bundle["records"]:
            for i in rec["idxs"]:
                owner[int(i)] = int(rec["anchor"])
        if mode == "ident" or rho_audio is None or mode == "video":
            lifted = np.ones(n_a, dtype=np.float64)
        else:
            lifted = _lift(rho_a, kept, n_a, owner)
        w1 = _match_total(bundle["w0_a"], bundle["w0_a"] * (1.0 + lifted))
        scores = bundle["attn_a"]
        records, _ = compress_tokens(
            bundle["raw_a"], scores, bundle["k_a"], "audio",
            acoustic=bundle["acoustic"], anchor_only=True, weights=w1,
        )
        anchors = [int(rec["anchor"]) for rec in records]
        drop = torch.ones(n_a, dtype=torch.bool)
        drop[anchors] = False
        keep[bundle["a_pos"][drop.to(device)]] = False
    chosen = []
    if n_v and bundle["raw_v"] is not None:
        kept = sorted(int(j) for j in bundle["chosen"])
        owner = video_owner(bundle["packed"], bundle["chosen"])
        if mode == "ident" or rho_video is None or mode == "audio":
            lifted_v = np.ones(n_v, dtype=np.float64)
        else:
            lifted_v = _lift(rho_v, kept, n_v, owner)
        w1 = _match_total(bundle["w0_v"], bundle["w0_v"] * (1.0 + lifted_v))
        chosen = select_video(
            bundle["raw_v"], bundle["attn_v"], bundle["coords"], bundle["k_v"],
            cross=True, use_attn=True, weights=w1, packed=bundle["packed"],
        )
        drop = torch.ones(n_v, dtype=torch.bool)
        drop[chosen] = False
        keep[bundle["v_pos"][drop.to(device)]] = False
    return keep, anchors, chosen


def probe_rho(text_model, inputs_embeds, attention_mask, position_ids, input_ids, audio_id, video_id):
    import torch
    from omnizip.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb, repeat_kv

    ids = input_ids[0]
    media = (ids == audio_id) | (ids == video_id)
    if int(media.sum()) == 0:
        return None, None, 0
    last_media = int(media.nonzero(as_tuple=True)[0][-1])
    index = torch.arange(ids.shape[0], device=ids.device)
    text_after = ((ids != audio_id) & (ids != video_id) & (index > last_media)).nonzero(as_tuple=True)[0]
    if text_after.numel() == 0:
        return None, None, 0
    qpos = text_after[-4:]
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(3, 1, -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    causal = text_model._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden = inputs_embeds
    for layer in text_model.layers[:3]:
        hidden = layer(
            hidden,
            attention_mask=causal,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    attn = text_model.layers[3].self_attn
    normed = text_model.layers[3].input_layernorm(hidden)
    bsz, q_len, _ = normed.shape
    query = attn.q_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(normed).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    query, key = apply_multimodal_rotary_pos_emb(query, key, position_embeddings[0], position_embeddings[1], attn.rope_scaling["mrope_section"])
    key = repeat_kv(key, attn.num_key_value_groups)
    query = query[:, :, qpos, :]
    logits = torch.matmul(query, key.transpose(2, 3)) / (attn.head_dim ** 0.5)
    for row, pos in enumerate(qpos.tolist()):
        logits[:, :, row, pos + 1:] = torch.finfo(logits.dtype).min
    if attention_mask is not None and attention_mask.dim() == 2:
        logits = logits.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(logits.dtype).min)
    alpha = torch.softmax(logits.float(), dim=-1)
    def density(token_id):
        pos = (ids == token_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return None
        mass = alpha[:, :, :, pos].sum(dim=(0, 1, 2))
        total = float(mass.sum())
        if total <= 0:
            return None
        p = (mass / total).detach().float().cpu().numpy()
        return p * len(p)
    return density(audio_id), density(video_id), int(qpos.numel())


def correct_keep_mask(
    text_model, input_ids, inputs_embeds, attention_mask, position_ids,
    input_features, feature_attention_mask, video_grid_thw,
    audio_token_id, video_token_id, keep_ratio, spatial_merge_size,
    audio_scores=None, video_scores=None, query_mode="both", attn_sec=0.0,
):
    import torch
    t0 = time.time()
    ids = input_ids[0]
    emb = inputs_embeds[0]
    bundle = prepare_draft(
        emb, ids, audio_token_id, video_token_id, keep_ratio, video_grid_thw, spatial_merge_size,
        audio_scores, video_scores, input_features, feature_attention_mask,
    )
    keep0 = bundle["keep0"]
    rho_a = rho_v = None
    n_query = 0
    probe_sec = 0.0
    if query_mode != "ident":
        t1 = time.time()
        sliced = keep0
        try:
            with torch.no_grad():
                rho_a, rho_v, n_query = probe_rho(
                    text_model,
                    inputs_embeds[:, sliced, :],
                    None if attention_mask is None else attention_mask[:, sliced],
                    None if position_ids is None else position_ids[:, :, sliced],
                    input_ids[:, sliced],
                    audio_token_id, video_token_id,
                )
        except Exception as exc:
            rho_a = rho_v = None
            n_query = 0
            print("omniquery probe failed", type(exc).__name__, exc, flush=True)
        probe_sec = time.time() - t1
        # probe_rho already returns densities in increasing token order.
    keep, anchors, chosen = reselect(bundle, rho_a, rho_v, query_mode)
    draft_a = sorted(int(rec["anchor"]) for rec in bundle["records"])
    draft_v = sorted(int(j) for j in bundle["chosen"])
    same_a = anchors == draft_a
    same_v = sorted(int(j) for j in chosen) == draft_v
    inter_a = len(set(anchors) & set(draft_a))
    inter_v = len(set(int(j) for j in chosen) & set(draft_v))
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "probe_sec": round(probe_sec, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": bundle["n_a"], "k_audio": len(anchors) if anchors else bundle["k_a"],
        "n_video": bundle["n_v"], "k_video": len(chosen) if chosen else bundle["k_v"],
        "query_mode": query_mode, "n_query": n_query,
        "overlap_audio": inter_a, "overlap_video": inter_v,
        "same": int(same_a and same_v),
    })
    print(
        "omniquery", query_mode,
        "A", LAST["k_audio"], "/", LAST["n_audio"],
        "V", LAST["k_video"], "/", LAST["n_video"],
        "overlap", inter_a, inter_v,
        "same", int(same_a and same_v),
        "q", n_query,
        "sec", LAST["sec"], "probe", LAST["probe_sec"],
        flush=True,
    )
    return keep


def _self_check():
    import torch
    from omnizip.omnianchor import compress_tokens
    from omnizip.omnifac import _cosine_edges, _covers, select_video

    raw = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    scores = np.array([0.4, 0.1, 0.4, 0.1])
    recs, _ = compress_tokens(raw, scores, 2, "audio", anchor_only=True)
    w0 = _base_weight(_attn(scores, 4))
    recs2, _ = compress_tokens(raw, scores, 2, "audio", anchor_only=True, weights=2 * w0)
    assert [r["anchor"] for r in recs] == [r["anchor"] for r in recs2]
    covered = sorted(i for rec in recs for i in rec["idxs"])
    assert covered == [0, 1, 2, 3]
    coords = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    packed = _cosine_edges(raw.numpy(), _covers(coords, True))
    a = select_video(raw.numpy(), scores, coords, 2, packed=packed)
    b = select_video(raw.numpy(), scores, coords, 2, weights=2 * _base_weight(_attn(scores, 4)), packed=packed)
    assert a == b, (a, b)
    owner = video_owner(packed, a)
    assert owner[a[0]] == a[0]
    print("omniquery self-check ok", [r["anchor"] for r in recs], a)


if __name__ == "__main__":
    _self_check()
