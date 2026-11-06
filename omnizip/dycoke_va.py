import torch

def _ttm_keep_indices(num_tokens, num_per, features, keep_ratio):
    nframes = num_tokens // num_per
    keep = []
    if nframes < 2 or num_per <= 0:
        return list(range(num_tokens))
    sims = []
    for i in range(0, nframes - 1, 2):
        a = features[i * num_per:(i + 1) * num_per]
        b = features[(i + 1) * num_per:(i + 2) * num_per]
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
        sims.append(torch.nn.functional.cosine_similarity(a, b, dim=1))
    # keep_ratio is the fraction of all tokens in this modality. The first frame of
    # each pair is kept, so the second frame keeps 2*keep_ratio-1. At 0.5 the
    # second frame is dropped and the modality keeps half its tokens.
    dynamic_ratio = min(1.0, max(0.0, 2.0 * float(keep_ratio) - 1.0))
    n_keep = int(round(dynamic_ratio * num_per))
    for i in range(0, nframes - 1, 2):
        keep.extend(range(i * num_per, (i + 1) * num_per))
        if n_keep > 0:
            idx = sims[i // 2].topk(min(n_keep, num_per), largest=False).indices
            base = (i + 1) * num_per
            keep.extend((base + idx).tolist())
    if nframes % 2 == 1:
        keep.extend(range((nframes - 1) * num_per, nframes * num_per))
    keep.extend(range(nframes * num_per, num_tokens))
    return keep

def _apply(mask, pos, local_keep):
    drop = torch.ones(pos.numel(), dtype=torch.bool, device=pos.device)
    drop[local_keep] = False
    mask[pos[drop]] = False

def dycoke_va(input_embeds, input_ids, audio_token_id, video_token_id, video_grid_thw, keep_ratio=0.5, audio_tokens_per_group=25):
    ids = input_ids[0]
    emb = input_embeds[0]
    device = emb.device
    mask = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    if v_pos.numel() and video_grid_thw is not None:
        tpf = int(video_grid_thw[0, 1].item() // 2) * int(video_grid_thw[0, 2].item() // 2)
        if tpf > 0:
            _apply(mask, v_pos, _ttm_keep_indices(v_pos.numel(), tpf, emb[v_pos], keep_ratio))
    if a_pos.numel() and audio_tokens_per_group > 0:
        _apply(mask, a_pos, _ttm_keep_indices(a_pos.numel(), audio_tokens_per_group, emb[a_pos], keep_ratio))
    return input_embeds, mask

def random_va(input_embeds, input_ids, audio_token_id, video_token_id, keep_ratio=0.55, generator=None):
    ids = input_ids[0]
    device = input_embeds.device
    mask = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    g = generator or torch.Generator(device="cpu")
    for token_id in (video_token_id, audio_token_id):
        pos = (ids == token_id).nonzero(as_tuple=True)[0]
        n = pos.numel()
        if n == 0:
            continue
        k = max(1, int(round(keep_ratio * n)))
        choice = torch.randperm(n, generator=g)[:k]
        drop = torch.ones(n, dtype=torch.bool, device=device)
        drop[choice] = False
        mask[pos[drop]] = False
    return input_embeds, mask
