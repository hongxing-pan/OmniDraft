"""Late entry of a fixed 30% token set.

The final tokens are chosen once. Full mode runs every selected token from
layer 0. Raw and delta modes run 15% plus all text through the first 4 layers,
then insert the remaining tokens at that boundary and continue. New tokens do
not receive KV in the first 4 layers.
"""
import time

import numpy as np

LAST = {}
SPLIT = 4


def _budgets(n, keep_ratio, minimum):
    full = int(max(minimum, min(n, round(keep_ratio * n)))) if n else 0
    early = int(max(minimum, min(full, round(0.5 * keep_ratio * n)))) if n else 0
    if full and early > full:
        early = full
    return early, full


def _select(emb, ids, audio_id, video_id, keep_ratio, video_grid_thw, spatial_merge_size, audio_scores, video_scores, input_features, feature_attention_mask):
    from omnizip.omnirefine import _cut, audio_tree, video_order
    from omnizip.omnispan import align_mel, video_coords
    a_pos = (ids == audio_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_id).nonzero(as_tuple=True)[0]
    n_a, n_v = int(a_pos.numel()), int(v_pos.numel())
    early_a, full_a = [], []
    if n_a:
        acoustic = None
        if input_features is not None:
            mel = input_features[0].detach().float().cpu()
            if feature_attention_mask is not None:
                mel = mel[:, feature_attention_mask[0].bool().cpu()]
            acoustic = align_mel(mel, n_a)
        nodes, _w, _raw = audio_tree(emb[a_pos].detach(), audio_scores, acoustic)
        e, f = _budgets(n_a, keep_ratio, 1)
        early_nodes = _cut(nodes, n_a, e)
        early_a = [int(node["anchor"]) for node in early_nodes]
        full_a = [int(node["anchor"]) for node in _cut(nodes, n_a, f)]
        cover = {}
        for node in early_nodes:
            for i in node["idxs"]:
                cover[int(i)] = int(node["anchor"])
    else:
        cover = {}
    early_v, full_v, packed = [], [], None
    if n_v and video_grid_thw is not None:
        raw = emb[v_pos].detach().float().cpu().numpy()
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(raw.shape[0], tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
        _e, f = _budgets(n_v, keep_ratio, n_frames)
        order, packed, _w = video_order(raw, None if video_scores is None else video_scores.detach().float().cpu().numpy(), coords, f)
        e = min(_e, len(order))
        early_v = [int(i) for i in order[:e]]
        full_v = [int(i) for i in order]
    return a_pos, v_pos, early_a, full_a, early_v, full_v, packed, cover


def _rep_audio(early, full, a_pos, cover):
    """Early-region anchor, unless that anchor sits after this token."""
    early_set = set(early)
    rep = {}
    for i in full:
        if i in early_set:
            continue
        anchor = cover.get(int(i))
        if anchor is None:
            continue
        if int(a_pos[anchor]) <= int(a_pos[i]):
            rep[i] = anchor
    return rep


def _rep_video(early, full, v_pos, packed):
    early_set = set(early)
    rep = {}
    if packed is None:
        return rep
    pos = {i: int(v_pos[i]) for i in full}
    for i in full:
        if i in early_set:
            continue
        best_j, best_s = None, -1.0
        for j in early:
            if pos[j] > pos[i]:
                continue
            nbrs, sim = packed[j]
            hit = np.where(nbrs == i)[0]
            if hit.size == 0:
                continue
            s = float(sim[int(hit[0])])
            if s > best_s:
                best_j, best_s = j, s
        rep[i] = best_j
    return rep


def _order(ids, audio_id, video_id, a_pos, v_pos, a_idx, v_idx):
    text = ((ids != audio_id) & (ids != video_id)).nonzero(as_tuple=True)[0]
    cols = [int(i) for i in text.tolist()]
    cols += [int(a_pos[i]) for i in a_idx]
    cols += [int(v_pos[i]) for i in v_idx]
    cols.sort()
    return cols


def _late_positions(early_cols, full_cols):
    early = set(early_cols)
    return [p for p in full_cols if p not in early]


def _reroute(text_model, h_late, pos_late, cache, early_cols, late_cols, layers):
    """Bring skipped tokens up to the insertion layer.

    Each skipped layer attends to keys already stored for the early tokens,
    plus earlier late tokens. The mask follows original sequence positions.
    """
    import torch
    n = h_late.shape[1]
    if n == 0:
        return h_late
    ne = len(early_cols)
    device = h_late.device
    early_pos = torch.tensor(early_cols, device=device)
    late_pos = torch.tensor(late_cols, device=device)
    allow_early = early_pos.view(1, ne) < late_pos.view(n, 1)
    allow_late = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    allow = torch.cat([allow_early, allow_late], dim=1)
    neg = torch.finfo(h_late.dtype).min
    mask = torch.zeros(1, 1, n, ne + n, device=device, dtype=h_late.dtype)
    mask = mask.masked_fill(~allow.view(1, 1, n, ne + n), neg)
    hidden = h_late
    for layer in layers:
        past_len = cache.get_seq_length(layer.self_attn.layer_idx)
        cache_position = torch.arange(past_len, past_len + n, device=device)
        position_embeddings = text_model.rotary_emb(hidden, pos_late)
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=pos_late,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    return hidden


def _catchup(h_late, h0_early, h4_early):
    """Affine map fit on tokens that ran the early layers."""
    src = h0_early.reshape(-1, h0_early.shape[-1]).float()
    dst = h4_early.reshape(-1, h4_early.shape[-1]).float()
    src_c = src - src.mean(dim=0, keepdim=True)
    dst_c = dst - dst.mean(dim=0, keepdim=True)
    var = src_c.pow(2).sum(dim=0).clamp(min=1e-6)
    scale = (src_c * dst_c).sum(dim=0) / var
    bias = dst.mean(dim=0) - scale * src.mean(dim=0)
    return (h_late.float() * scale + bias).to(dtype=h_late.dtype)


def _run(text_model, hidden, position_ids, layers, cache, past_len):
    import torch
    seq = hidden.shape[1]
    cache_position = torch.arange(past_len, past_len + seq, device=hidden.device)
    position_embeddings = text_model.rotary_emb(hidden, position_ids)
    # Prefill uses a square causal mask. Decode attends to the whole cache.
    # None lets SDPA use is_causal on a prefill and attend to the whole cache on a single new token.
    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    return hidden


def prefill_hidden(
    thinker, input_ids, inputs_embeds, position_ids,
    input_features, feature_attention_mask, video_grid_thw,
    audio_scores, video_scores, keep_ratio, late_mode,
):
    import torch
    from transformers.cache_utils import DynamicCache
    t0 = time.perf_counter()
    old = getattr(thinker, "_omnilate_state", None)
    if isinstance(old, dict):
        old["cache"] = None
    thinker._omnilate_state = None
    text_model = thinker.model
    ids = input_ids[0]
    emb = inputs_embeds
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    a_pos, v_pos, early_a, full_a, early_v, full_v, packed, cover = _select(
        emb[0], ids, thinker.config.audio_token_id, thinker.config.video_token_id,
        keep_ratio, video_grid_thw, int(thinker.visual.spatial_merge_size),
        audio_scores, video_scores, input_features, feature_attention_mask,
    )
    early_cols = _order(ids, thinker.config.audio_token_id, thinker.config.video_token_id, a_pos, v_pos, early_a, early_v)
    full_cols = _order(ids, thinker.config.audio_token_id, thinker.config.video_token_id, a_pos, v_pos, full_a, full_v)
    early_index = torch.tensor(early_cols, device=emb.device)
    full_index = torch.tensor(full_cols, device=emb.device)
    pos_full = position_ids[:, :, full_index]
    h_full = emb[:, full_index]
    layers = list(text_model.layers)
    cache = DynamicCache()
    mode = late_mode if late_mode in ("full", "raw", "delta") else "raw"
    t_sel = time.perf_counter()
    if mode == "full":
        hidden = _run(text_model, h_full, pos_full, layers, cache, 0)
        boundary = None
    else:
        pos_early = position_ids[:, :, early_index]
        h_early = emb[:, early_index]
        h4 = _run(text_model, h_early, pos_early, layers[:SPLIT], cache, 0)
        t_early = time.perf_counter()
        late_cols = _late_positions(early_cols, full_cols)
        if late_cols:
            late_index = torch.tensor(late_cols, device=emb.device)
            h_late = _reroute(
                text_model, emb[:, late_index], position_ids[:, :, late_index],
                cache, early_cols, late_cols, layers[:SPLIT],
            )
            if mode == "delta":
                early_set = set(early_cols)
                keep = [i for i, p in enumerate(full_cols) if p in early_set]
                h_late = _catchup(h_late, h_full[:, keep], h4)
        else:
            h_late = None
        hidden = _merge(h_full, h4, early_cols, full_cols, a_pos, v_pos, early_a, full_a, early_v, full_v, packed, cover, mode)
        if h_late is not None:
            late_of = {p: i for i, p in enumerate(late_cols)}
            for i, p in enumerate(full_cols):
                if p in late_of:
                    hidden[0, i] = h_late[0, late_of[p]]
        hidden = _run(text_model, hidden, pos_full, layers[SPLIT:], cache, 0)
        boundary = t_early
    hidden = text_model.norm(hidden)
    thinker._omnilate_state = {
        "cache": cache,
        "early_len": len(early_cols) if mode != "full" else len(full_cols),
        "full_len": len(full_cols),
        "next_pos": pos_full[:, :, -1:] + 1,
        "mode": mode,
    }
    LAST.clear()
    LAST.update({
        "mode": mode,
        "early": len(early_cols),
        "full": len(full_cols),
        "select_sec": round(t_sel - t0, 3),
        "early_sec": None if boundary is None else round(boundary - t_sel, 3),
        "rest_sec": round(time.perf_counter() - (boundary or t_sel), 3),
    })
    print("omnilate", mode, "early", LAST["early"], "full", LAST["full"], "sel", LAST["select_sec"], "early_sec", LAST["early_sec"], "rest", LAST["rest_sec"], flush=True)
    return hidden


def _merge(h_full, h4, early_cols, full_cols, a_pos, v_pos, early_a, full_a, early_v, full_v, packed, cover, mode):
    import torch
    early_of = {p: i for i, p in enumerate(early_cols)}
    rep_a = _rep_audio(early_a, full_a, a_pos, cover)
    rep_v = _rep_video(early_v, full_v, v_pos, packed)
    seq_rep = {}
    for local, rep in rep_a.items():
        if rep is not None:
            seq_rep[int(a_pos[local])] = int(a_pos[rep])
    for local, rep in rep_v.items():
        if rep is not None:
            seq_rep[int(v_pos[local])] = int(v_pos[rep])
    out = h_full.clone()
    h4_by = {p: h4[0, i] for i, p in enumerate(early_cols)}
    h0_by = {p: h_full[0, i] for i, p in enumerate(full_cols)}
    borrowed = 0
    for i, p in enumerate(full_cols):
        if p in early_of:
            out[0, i] = h4_by[p]
            continue
        if mode == "delta" and seq_rep.get(p) in h4_by:
            r = seq_rep[p]
            out[0, i] = h0_by[p] + (h4_by[r] - h0_by[r])
            borrowed += 1
        # raw, and delta without a legal rep, keep h_full which is h^(0)
    LAST["borrowed"] = borrowed
    LAST["late"] = len(full_cols) - len(early_cols)
    return out


def greedy_continue(thinker, first_logits, max_new_tokens, eos_id):
    import torch
    state = thinker._omnilate_state
    logits = first_logits[0, -1]
    ids = []
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits).item())
        if not ids:
            print("omnilate first", nxt, flush=True)
        ids.append(nxt)
        if eos_id is not None and nxt == int(eos_id):
            break
        logits = _decode_one(thinker, nxt, state)
    LAST["decode_sec"] = round(time.perf_counter() - t0, 3)
    print("omnilate decode", LAST["decode_sec"], "tokens", len(ids), flush=True)
    return ids


def _decode_one(thinker, token_id, state):
    import torch
    text_model = thinker.model
    device = state["next_pos"].device
    emb = text_model.embed_tokens(torch.tensor([[token_id]], device=device))
    pos = state["next_pos"]
    layers = list(text_model.layers)
    cache = state["cache"]
    if state["mode"] == "full":
        hidden = _run(text_model, emb, pos, layers, cache, state["full_len"])
        state["full_len"] += 1
    else:
        hidden = _run(text_model, emb, pos, layers[:SPLIT], cache, state["early_len"])
        state["early_len"] += 1
        hidden = _run(text_model, hidden, pos, layers[SPLIT:], cache, state["full_len"])
        state["full_len"] += 1
    state["next_pos"] = pos + 1
    hidden = text_model.norm(hidden)
    return thinker.lm_head(hidden[:, -1])[0]
