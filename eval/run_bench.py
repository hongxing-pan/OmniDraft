import time
import argparse, json, os, sys, gc
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen-omni-utils" / "src"))
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def letter(s):
    for ch in (s or ""):
        if ch in "ABCD":
            return ch
    return ""

def load_samples(dataset, limit, offset=0):
    base = os.environ.get("OSELECT_DATA", "/HDDDATA/phx/tmp/omniselect/data")
    out = []
    if dataset == "daily":
        if os.environ.get("OSELECT_FULL") == "1":
            daily = f"{base}/DailyOmni/qa.json"
        else:
            daily = f"{base}/pack1000/daily_qa.json"
        rows = json.load(open(daily if os.path.exists(daily) else f"{base}/DailyOmni/qa.json"))
        rows = rows[offset:offset + limit]
        for i, r in enumerate(rows, start=offset):
            vid = r["video_id"]
            out.append({
                "key": f"{vid}_{i}",
                "video": f"{base}/DailyOmni/Videos/{vid}/{vid}_video.mp4",
                "question": r["Question"],
                "options": r["Choice"],
                "gt": letter(r["Answer"]),
            })
    elif dataset == "ovb":
        ovb = f"{base}/pack1000/ovb_data.json"
        rows = json.load(open(ovb if os.path.exists(ovb) else f"{base}/OmniVideoBench/data.json"))[:limit]
        for i, r in enumerate(rows):
            rel = r["video"]
            out.append({
                "key": f"video_{i}",
                "video": f"{base}/OmniVideoBench/{rel}",
                "question": r["question"],
                "options": r["options"],
                "gt": letter(r.get("correct_option") or r.get("answer") or ""),
            })
    elif dataset == "vme":
        pack = f"{base}/pack1000/videomme_1000.json"
        parquet = f"{base}/VideoMME/videomme/test-00000-of-00001.parquet"
        if os.environ.get("OSELECT_FULL") == "1":
            import pyarrow.parquet as pq
            rows = pq.read_table(parquet).to_pylist()
            rows = rows[offset:offset + limit]
        elif os.path.exists(pack):
            rows = json.load(open(pack))
            rows = rows[offset:offset + limit]
        else:
            import pyarrow.parquet as pq
            rows = pq.read_table(parquet).to_pylist()
            rows = rows[offset:offset + limit]
        for i, r in enumerate(rows, start=offset):
            vid = r["videoID"]
            opts = r["options"]
            if isinstance(opts, str):
                try:
                    opts = json.loads(opts)
                except Exception:
                    pass
            out.append({
                "key": f"{vid}_{i}",
                "video": f"{base}/VideoMME/videos/data/{vid}.mp4",
                "question": r["question"],
                "options": opts,
                "gt": letter(r.get("answer") or r.get("gt_answer") or ""),
            })
    elif dataset == "ws":
        qa = json.load(open(f"{base}/WorldSense/worldsense_qa.json"))
        rows = []
        for v in qa.values():
            for key, task in v.items():
                if not str(key).startswith("task"):
                    continue
                rows.append((v, key, task))
        for v, key, task in rows[offset:offset + limit]:
            vid = v["video_id"]
            out.append({
                "key": f"{vid}_{key}",
                "video": f"{base}/WorldSense/videos/{vid}.mp4",
                "question": task["question"],
                "options": task["candidates"],
                "gt": letter(task["answer"]),
            })
    else:
        raise SystemExit(dataset)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["daily", "ovb", "vme", "ws"], required=True)
    ap.add_argument("--method", choices=["omnizip", "omniselect", "dycoke", "random", "omnspan", "omnanchor", "omniread", "omnifac", "omnijoin", "omnipair", "omniwitness", "omnimmd", "omniquery", "omnirefine", "omnilate", "omnidraft", "full"], required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--nframes", type=int, default=128)
    ap.add_argument("--rho", type=float, default=0.55)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--audio_mode", default="span")
    ap.add_argument("--video_mode", default="span")
    ap.add_argument("--anchor_mode", choices=["full", "noattn", "anchor"], default="full")
    ap.add_argument("--compress", choices=["both", "audio", "video"], default="both")
    ap.add_argument("--read_mode", choices=["usage", "mass", "feat", "err"], default="usage")
    ap.add_argument("--budget_mode", choices=["shared", "fixed"], default="shared")
    ap.add_argument("--emit", choices=["merge", "rep", "center"], default="merge")
    ap.add_argument("--fac_mode", choices=["full", "spatial", "noattn"], default="full")
    ap.add_argument("--pair_mode", choices=["indep", "joint", "shuffle"], default="joint")
    ap.add_argument("--witness_mode", choices=["pair", "point"], default="pair")
    ap.add_argument("--query_mode", choices=["ident", "audio", "video", "both"], default="both")
    ap.add_argument("--refine_mode", choices=["grad", "shuffle", "geom"], default="grad")
    ap.add_argument("--late_mode", choices=["full", "raw", "delta"], default="raw")
    ap.add_argument("--exec", choices=["rerun", "splice", "twoselect", "twofuse"], default="rerun")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--vis", default="flash")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--round_l1", type=int, default=4)
    ap.add_argument("--boost", default="")
    ap.add_argument("--signal", default="quota")
    ap.add_argument("--null_sub", type=int, default=1)
    ap.add_argument("--null_text", default="What is happening in the video?")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--draft_ratio", type=float, default=0.05)
    ap.add_argument("--probe_layer", type=int, default=12)
    ap.add_argument("--splice_l", type=int, default=12)
    ap.add_argument("--splice_r", type=float, default=0.0)
    ap.add_argument("--gate", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="cuda:0", attn_implementation="flash_attention_2"
    )
    cfg = {"rho_audio": args.rho, "rho_video": args.rho, "g": 3, "contextual_ratio": 0.0, "keep_ratio": args.keep_ratio, "method": args.method, "audio_mode": args.audio_mode, "video_mode": args.video_mode, "anchor_mode": args.anchor_mode, "compress": args.compress, "read_mode": args.read_mode, "budget_mode": args.budget_mode, "emit": args.emit, "fac_mode": args.fac_mode, "pair_mode": args.pair_mode, "witness_mode": args.witness_mode, "query_mode": args.query_mode, "refine_mode": args.refine_mode, "late_mode": args.late_mode}
    model.thinker.omnizip_config = cfg
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    if args.method == "omnidraft":
        model.thinker.omnizip_config.update({
            "exec": args.exec, "lam": args.lam, "draft_ratio": args.draft_ratio,
            "probe_layer": args.probe_layer, "splice_l": args.splice_l, "splice_r": args.splice_r,
            "gate": args.gate, "signal": args.signal, "boost": args.boost, "null_sub": args.null_sub, "vis": args.vis, "rounds": args.rounds, "round_l1": args.round_l1,
            "null_question_ids": processor.tokenizer(args.null_text, add_special_tokens=False).input_ids,
            "special_ids": list(processor.tokenizer.all_special_ids),
            "im_end_id": processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        })

    _opt = []
    for _ch in "ABCDEFGH":
        _tok = processor.tokenizer.encode(_ch, add_special_tokens=False)
        _opt.append(int(_tok[0]) if len(_tok) == 1 else -1)
    model.thinker.omnizip_config["option_token_ids"] = _opt
    samples = load_samples(args.dataset, args.limit, args.offset)
    pixels = args.nframes * 128 * 28 * 28
    correct = scored = 0
    for i, sample in enumerate(samples):
        save = os.path.join(args.output_dir, f"{sample['key']}.json")
        if os.path.exists(save):
            o = json.load(open(save))
            if not o.get("oom") and not o.get("fail"):
                scored += 1
                correct += int(bool(o.get("is_correct")))
            continue
        opts = sample["options"]
        if isinstance(opts, list):
            opts = "\n".join(str(x) for x in opts)
        prompt = f"{sample['question']}\nOptions:\n{opts}\nAnswer with the option's letter from the given choices directly."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [
                {"type": "video", "video": sample["video"], "nframes": args.nframes, "min_pixels": 128 * 28 * 28, "max_pixels": 768 * 28 * 28, "total_pixels": pixels},
                {"type": "text", "text": prompt},
            ]},
        ]
        try:
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            model.thinker.nframes = videos[0].shape[0]
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(model.device).to(model.dtype)
            if args.method == "omniselect":
                from omnizip.omniselect_proto import prepare_need
                model.thinker._omniselect_need = prepare_need(
                    sample["question"], videos, audios, inputs["input_ids"],
                    model.thinker.config.audio_token_id, model.thinker.config.video_token_id,
                    args.keep_ratio,
                )
            if args.method == "omnidraft" and args.exec in ("splice", "twofuse"):
                _tgen = time.perf_counter()
                with torch.no_grad():
                    out = model.thinker(**inputs, use_audio_in_video=True, return_dict=True, use_cache=False)
                    from omnizip.omnidraft import greedy_splice
                    new_ids = greedy_splice(model.thinker, out.logits, 8, processor.tokenizer.eos_token_id)
                print("question_sec", round(time.perf_counter() - _tgen, 3), flush=True)
                resp = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
                cont = new_ids
            elif args.method == "omnilate":
                with torch.no_grad():
                    out = model.thinker(**inputs, use_audio_in_video=True, return_dict=True, use_cache=False)
                    from omnizip.omnilate import greedy_continue
                    new_ids = greedy_continue(model.thinker, out.logits, 8, processor.tokenizer.eos_token_id)
                resp = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
                cont = new_ids
            else:
                _tgen=time.perf_counter()
                cont = model.generate(**inputs, return_audio=False, do_sample=False, thinker_do_sample=False, max_new_tokens=8, use_audio_in_video=True)
                print("question_sec", round(time.perf_counter() - _tgen, 3), flush=True)
                if os.environ.get("OSELECT_TIME"):
                    print("generate_sec", round(time.perf_counter()-_tgen, 3), flush=True)
                resp = processor.tokenizer.decode(cont[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = letter(resp)
            ok = pred == sample["gt"] and pred != ""
            stats=getattr(model.thinker,"_omnspan_last",None)
            json.dump({"key": sample["key"], "gt": sample["gt"], "prediction": resp, "is_correct": ok, "selector": stats}, open(save, "w"), ensure_ascii=False)
            scored += 1
            correct += int(ok)
            print(f"{i} pred {pred} gt {sample['gt']} acc {correct}/{scored}", flush=True)
            del inputs, cont
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache(); gc.collect()
            json.dump({"key": sample["key"], "oom": True, "err": str(e)[:180]}, open(save, "w"))
            print(f"{i} OOM", flush=True)
        except ValueError as e:
            if "nframes" in str(e) and "nframes" in conversation[1]["content"][0]:
                conversation[1]["content"][0].pop("nframes", None)
                try:
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
                    model.thinker.nframes = videos[0].shape[0]
                    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                    inputs = inputs.to(model.device).to(model.dtype)
                    cont = model.generate(**inputs, return_audio=False, do_sample=False, thinker_do_sample=False, max_new_tokens=8, use_audio_in_video=True)
                    resp = processor.tokenizer.decode(cont[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    pred = letter(resp)
                    ok = pred == sample["gt"] and pred != ""
                    json.dump({"key": sample["key"], "gt": sample["gt"], "prediction": resp, "is_correct": ok, "short": True}, open(save, "w"), ensure_ascii=False)
                    scored += 1
                    correct += int(ok)
                    print("{0} short pred {1} gt {2} acc {3}/{4}".format(i, pred, sample["gt"], correct, scored), flush=True)
                    del inputs, cont
                    torch.cuda.empty_cache()
                    continue
                except torch.cuda.OutOfMemoryError as e2:
                    torch.cuda.empty_cache(); gc.collect()
                    json.dump({"key": sample["key"], "oom": True}, open(save, "w"))
                    print(f"{i} OOM", flush=True)
                    continue
            json.dump({"key": sample["key"], "error": str(e)[:180], "is_correct": False}, open(save, "w"))
            print(f"{i} ERR {e}", flush=True)
        except Exception as e:
            json.dump({"key": sample["key"], "fail": True, "err": str(e)[:180]}, open(save, "w"))
            print(f"{i} FAIL {type(e).__name__}", flush=True)
    print(f"DONE scored {scored} correct {correct}", flush=True)

if __name__ == "__main__":
    main()
