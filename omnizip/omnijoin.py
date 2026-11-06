"""Audio JS merge plus video facility location, fixed per-modality budgets.

Audio merges only time-adjacent readership segments and writes the
readership-weighted mean at the max-p original token. Video keeps original
tokens chosen by the existing short-range spatiotemporal facility location.
"""
import time

import numpy as np

LAST = {}


def omnijoin_keep_mask(
    input_ids, inputs_embeds, video_grid_thw, audio_token_id, video_token_id,
    keep_ratio, spatial_merge_size, audio_mass=None, audio_seg=None,
    video_scores=None, attn_sec=0.0,
):
    from omnizip.omnifac import select_video
    from omnizip.omniread import compress_modality, _write
    from omnizip.omnispan import video_coords
    t0 = time.time()
    ids = input_ids[0]
    device = inputs_embeds.device
    a_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    v_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    import torch
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=device)
    n_a = int(a_pos.numel())
    n_v = int(v_pos.numel())
    raw_a = inputs_embeds[0, a_pos].detach().float().cpu().numpy() if n_a else np.zeros((0, 1), np.float32)
    n_frames = 0
    coords = None
    if n_v and video_grid_thw is not None:
        tt, gh, gw = [int(x) for x in video_grid_thw[0].tolist()]
        coords = video_coords(n_v, tt, gh, gw, spatial_merge_size).numpy()
        n_frames = int(len(set(coords[:, 0].astype(int).tolist())))
    if n_a and (audio_mass is None or len(audio_mass) != n_a):
        raise RuntimeError(f"audio usage {0 if audio_mass is None else len(audio_mass)} != {n_a}")
    k_a = int(max(1, min(n_a, round(keep_ratio * n_a)))) if n_a else 0
    k_v = int(max(1, min(n_v, round(keep_ratio * n_v)))) if n_v else 0
    clamped = False
    if n_v and k_v < n_frames:
        clamped = True
        k_v = n_frames
    if n_a:
        audio_mass = np.asarray(audio_mass, dtype=np.float32)
        audio_seg = np.asarray(audio_seg, dtype=np.int32)
        rec_a, k_a = compress_modality(raw_a, audio_mass, audio_seg, k_a, "audio", "usage", center=False)
        _write(inputs_embeds, a_pos, rec_a, keep, "merge")
    if n_v and coords is not None:
        vv = inputs_embeds[0, v_pos].detach().float().cpu().numpy()
        scores = None if video_scores is None else video_scores.detach().float().cpu().numpy()
        chosen = select_video(vv, scores, coords, k_v, cross=True, use_attn=True)
        k_v = len(chosen)
        if v_pos.numel():
            drop = torch.ones(v_pos.shape[0], dtype=torch.bool)
            if chosen:
                drop[chosen] = False
            keep[v_pos[drop.to(device)]] = False
    LAST.clear()
    LAST.update({
        "sec": round(time.time() - t0, 3),
        "attn_sec": round(float(attn_sec), 3),
        "n_audio": n_a, "k_audio": k_a, "n_video": n_v, "k_video": k_v,
        "video_clamped": clamped, "n_frames": n_frames,
    })
    print(
        "omnijoin", "A", k_a, "/", n_a, "V", k_v, "/", n_v,
        "sec", LAST["sec"], "attn", LAST["attn_sec"], "clamp", int(clamped),
        flush=True,
    )
    return keep
