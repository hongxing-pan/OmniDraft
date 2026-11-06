"""OmniSelect under this repo's eval protocol.

Selection is the original group prune. Frames, pixels, prompt, and scoring
come from eval/run_bench.py. AudioCLIP scores are computed on the frames and
waveform that protocol already sampled.
"""
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

def allocate_pruning_weights(scores, num, total_pruning_ratio):
    device = scores.device
    num_tensor = torch.as_tensor(num, device=device, dtype=torch.float32)

    total_tokens = num_tensor.sum()
    total_expected_prune = total_tokens * total_pruning_ratio
    
    scores_std = scores.std()
    if scores_std < 1e-8:
        base_prune_probs = torch.full_like(scores, total_pruning_ratio)
    else:
        norm_scores = (scores - scores.mean()) / (scores_std + 1e-8)
        temperature = 1.0
        base_prune_probs = torch.sigmoid(-norm_scores / temperature)

    current_prune_sum = (base_prune_probs * num_tensor).sum()
    scale_factor = total_expected_prune / (current_prune_sum + 1e-8)
    block_prune_ratios = torch.clamp(base_prune_probs * scale_factor, 0.0, 1.0)

    for _ in range(5):
        current_sum = (block_prune_ratios * num_tensor).sum()
        diff = total_expected_prune - current_sum
        
        if abs(diff) < 1e-3:
            break
    
        adjustable_mask = (block_prune_ratios > 0) & (block_prune_ratios < 1)
        
        if not adjustable_mask.any() or num_tensor[adjustable_mask].sum() < 1e-6:
            break
            
        adjustment = diff / num_tensor[adjustable_mask].sum()
        block_prune_ratios[adjustable_mask] += adjustment
        block_prune_ratios = torch.clamp(block_prune_ratios, 0.0, 1.0)

    return block_prune_ratios

def guide_modal_prune(tmp_list,attn_mean,prune_ratio,global_mask,modality):
    device = attn_mean.device
    num_token = len(tmp_list)
    n_score = int(attn_mean.numel())
    if num_token == 0 or n_score == 0:
        return global_mask
    keep_num = int(max(1, num_token * (1.0 - prune_ratio)))
    keep_num = min(keep_num, n_score, num_token)

    if modality == 'video':
         _, topk_keep_indices = torch.topk(-attn_mean, keep_num, largest=True, sorted=False)
    else:
        _, topk_keep_indices = torch.topk(attn_mean, keep_num, largest=True, sorted=False)
    
    keep_mask = torch.zeros(num_token, dtype=torch.bool, device=device)
    keep_mask[topk_keep_indices] = True
    global_mask[tmp_list] = keep_mask
    
    return global_mask


@torch.no_grad()
def omniselect(
    video_grid_thw,
    input_embeds,
    audio_features,
    vision_features,
    attn_mean_a,
    attn_mean_v,
    input_ids,
    prune_need
):
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids
    
    merge_ratio_a = prune_need['args'].prune_ratio_a
    merge_ratio_v = prune_need['args'].prune_ratio_v
    
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    token_ids2group = prune_need['token_id2group']
    logits_a = prune_need['logits_a']
    logits_v = prune_need['logits_v']
    v_lst, a_lst = prune_need['v_lst'], prune_need['a_lst'] 
    cnt = 0
    num_v = []
    num_a = []
    mp = defaultdict(str)
    
    for i in range(len(v_lst)-1):
        cnt += 1
        mp[v_lst[i]] = 'v'
        mp[v_lst[i+1]] = 'v'
        if v_lst[i+1] - v_lst[i] != 1:
            num_v.append(cnt)
            cnt = 0
    num_v.append(cnt+1)
    cnt = 0
    
    for i in range(len(a_lst)-1):
        cnt += 1
        mp[a_lst[i]] = 'a'
        mp[a_lst[i+1]] = 'a'
        if a_lst[i+1] - a_lst[i] != 1:
            num_a.append(cnt)
            cnt = 0
    num_a.append(cnt+1)
    
    if prune_need['video_first'] == True: 
        if len(logits_v) % 4 != 0:
            k = len(logits_v) // 4
            main = logits_v[:k * 4].view(-1, 4)   # [k, 4]
            main_avg = main.mean(dim=1)  
            tail_avg = logits_v[k * 4:].mean().unsqueeze(0)
            pooled_v = torch.cat([main_avg, tail_avg])
        else:
            if len(logits_v) // 4 == len(num_v):
                pooled_v = torch.mean(logits_v.view(-1, 4),dim = 1)
            else:
                k = len(num_v) - 1
                main = logits_v[:k * 4].view(-1, 4) 
                main_avg = main.mean(dim=1)  
                tail_avg = logits_v[k * 4:].mean().unsqueeze(0)
                pooled_v = torch.cat([main_avg, tail_avg])
        
        weights = allocate_pruning_weights(pooled_v,num_v,total_pruning_ratio=merge_ratio_v)
        print("video first, pruning ratio are: ",weights)
        
        cur_timegroup = None
        cur_token_l_v = 0
        cur_token_l_a = 0
        tmp_audio = []
        tmp_video = []
        
        for i in range(len(input_ids)): 
            if mp[i] == 'v':
                tmp_video.append(i)
            elif mp[i] == 'a':
                tmp_audio.append(i)
            else:
                continue

            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune( 
                        tmp_video[:-1], 
                        attn_tmp_v,
                        weights[cur_timegroup - 1],
                        global_mask,
                        'video') 
                    
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        merge_ratio_a,
                        global_mask,
                        'audio')        
                    
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune( 
                tmp_video, 
                attn_tmp_v, 
                weights[cur_timegroup - 1],
                global_mask,
                'video')
        
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                merge_ratio_a,
                global_mask,
                'audio')
        
    elif prune_need['video_first'] == False: 
        if len(logits_a) > len(logits_v):
            prefix = logits_a[:-2]
            last_avg = logits_a[-2:].mean(dim=0, keepdim=True)
            logits_a = torch.cat([prefix, last_avg], dim=0)
        elif len(logits_a) % 4 != 0:
            k = len(logits_a) // 4
            r = len(logits_a) % 4
            main = logits_a[:k * 4].view(-1, 4)   # [4, k]
            main_avg = main.mean(dim=1)  
            tail_avg = logits_a[k * 4:].mean().unsqueeze(0)
            pooled_a = torch.cat([main_avg, tail_avg])
        else:
            if len(logits_a) // 4 == len(num_a):
                pooled_a = torch.mean(logits_a.view(-1, 4),dim = 1)
            else:
                k = len(num_v) - 1
                main = logits_a[:k * 4].view(-1, 4)   # [k, 4]
                main_avg = main.mean(dim=1)  
                tail_avg = logits_a[k * 4:].mean().unsqueeze(0)
                if len(num_a) > k:
                   pooled_a = torch.cat([main_avg, tail_avg])
                else:
                   main_avg[-1] = (main_avg[-1] + tail_avg[-1]) / 2.0
                   pooled_a = main_avg
                
        
        weights = allocate_pruning_weights(pooled_a,num_a,total_pruning_ratio=merge_ratio_a) 
        print('audio first, pruning ratio are:',weights)
        cur_timegroup = None
        tmp_audio = []
        tmp_video = []
        cur_token_l_v = 0
        cur_token_l_a = 0
        
        for i in range(len(input_ids)): 
            if mp[i] == 'a':
                tmp_audio.append(i)
            elif mp[i] == 'v':
                tmp_video.append(i)
            else:
                continue
            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        weights[cur_timegroup - 1],
                        global_mask,
                        'audio') 
                    
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune(
                        tmp_video[:-1],
                        attn_tmp_v,
                        merge_ratio_v,
                        global_mask,
                        'video')
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)           
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                weights[cur_timegroup - 1],
                global_mask,
                'audio') 
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune(
                tmp_video,
                attn_tmp_v,
                merge_ratio_v,
                global_mask,
                'video')          
    else:
        print("uniform pruning...")
        weights = [merge_ratio_v] * len(num_v) 
        cur_timegroup = None
        cur_token_l_v = 0
        cur_token_l_a = 0
        tmp_audio = []
        tmp_video = []
        
        for i in range(len(input_ids)): 
            if mp[i] == 'a':
                tmp_audio.append(i)
            elif mp[i] == 'v':
                tmp_video.append(i)
            else:
                continue

            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune( 
                        tmp_video[:-1], 
                        attn_tmp_v,
                        merge_ratio_v,
                        global_mask,
                        'video') 
                    
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        merge_ratio_a,
                        global_mask,
                        'audio')        
                   
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune( 
                tmp_video, 
                attn_tmp_v, 
                merge_ratio_v,
                global_mask,
                'video')
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                merge_ratio_a,
                global_mask,
                'audio')
            
    return global_mask


_CLIP = None


def _clip():
    global _CLIP
    if _CLIP is None:
        sys.path.append("/HDDDATA/phx/tmp/omniselect/OmniSelect/AudioCLIP")
        from model import AudioCLIP
        from utils_a.transforms import ToTensor1D
        import torchvision as tv
        from PIL import Image
        aclp = AudioCLIP(pretrained="/HDDDATA/phx/tmp/omniselect/weights/AudioCLIP-Full-Training.pt")
        device = torch.device("cuda:0")
        aclp.to(device).eval()
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        image_tf = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.Normalize(mean, std),
        ])
        _CLIP = (aclp, device, image_tf, ToTensor1D())
    return _CLIP


def prepare_need(question, videos, audios, input_ids, audio_token_id, video_token_id, keep_ratio, theta=2.0):
    """Build the dict omniselect() expects, from this protocol's tensors."""
    from PIL import Image
    from nltk import word_tokenize, pos_tag
    aclp, device, image_tf, _to1d = _clip()
    aclp.to(device)
    video = videos[0]
    if video.shape[1] == 3:
        visual = video.permute(0, 2, 3, 1)
    else:
        visual = video
    visual = visual.detach().float().cpu().numpy()
    if visual.max() <= 1.5:
        visual = visual * 255.0
    visual = visual.astype("uint8")
    nframes = int(visual.shape[0])
    wave = audios[0]
    if isinstance(wave, torch.Tensor):
        wave = wave.detach().float().cpu().numpy()
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size < nframes:
        wave = np.pad(wave, (0, nframes - wave.size))
    chunk = max(1, wave.size // nframes)
    audio_chunks = torch.from_numpy(wave[: nframes * chunk].reshape(nframes, 1, chunk))
    images = torch.stack([image_tf(Image.fromarray(fr)) for fr in visual])
    tokens = word_tokenize(question)
    tags = pos_tag(tokens)
    keywords = [w for w, t in tags if t.startswith("NN") or t.startswith("JJ")]
    text_input = keywords if keywords else tokens
    def _batched(tensor, forward, batch=32):
        parts = []
        for start in range(0, int(tensor.shape[0]), batch):
            piece = tensor[start:start + batch].to(device, non_blocking=True)
            parts.append(forward(piece).detach().cpu())
            del piece
        return torch.cat(parts, dim=0)

    with torch.no_grad():
        audio_features = _batched(
            audio_chunks, lambda x: aclp(audio=x)[0][0][0])
        image_features = _batched(
            images, lambda x: aclp(image=x)[0][0][1])
        ((_, _, text_features), _), _ = aclp(text=text_input)
        text_features = text_features.detach().cpu()
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)
        scale_at = float(torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0))
        scale_it = float(torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0))
        logits_audio = (scale_at * audio_features @ text_features.T).mean(dim=-1)
        logits_image = (scale_it * image_features @ text_features.T).mean(dim=-1)
        a_score = logits_audio.mean()
        v_score = logits_image.mean()
        logits_audio = logits_audio.detach().cpu()
        logits_image = logits_image.detach().cpu()
        a_score = a_score.detach()
        v_score = v_score.detach()
    del audio_chunks, images, audio_features, image_features, text_features
    aclp.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if abs(int(v_score) - int(a_score)) <= theta or len(keywords) == 0:
        a_score = v_score
    if a_score < v_score:
        video_first = True
    elif a_score > v_score:
        video_first = False
    else:
        video_first = "uniform"
    ids = input_ids[0]
    v_lst = (ids == video_token_id).nonzero(as_tuple=True)[0].tolist()
    a_lst = (ids == audio_token_id).nonzero(as_tuple=True)[0].tolist()
    groups = {}
    chunk_i = 0
    for lst in (v_lst, a_lst):
        chunk_i = 0
        for i in range(len(lst) - 1):
            groups[lst[i]] = chunk_i + 1
            if lst[i + 1] - lst[i] != 1:
                chunk_i += 1
        if lst:
            groups[lst[-1]] = chunk_i + 1
    prune = 1.0 - float(keep_ratio)
    args = SimpleNamespace(prune=True, prune_ratio_a=prune, prune_ratio_v=prune, theta=float(theta))
    return {
        "logits_a": logits_audio.detach(),
        "logits_v": logits_image.detach(),
        "video_first": video_first,
        "args": args,
        "token_id2group": groups,
        "v_lst": v_lst,
        "a_lst": a_lst,
    }
