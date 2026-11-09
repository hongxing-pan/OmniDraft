"""Efficiency table for OmniDraft_实验表格.tex, section 10 of the plan.

30 fixed questions per benchmark, 3 warmup questions outside the list, 3 repeats
per method, batch size 1. Every request is independent: the media is decoded
again for each repeat and neither AudioCLIP features nor draft selections are
carried over, so the costs the plan asks for are actually inside the timers.

  GPU Mem.      peak allocated during the request (weights stay resident)
  LLM Prefill   the final answer pass only, one full-length thinker text forward
  Latency       inputs on device -> answer text, includes encoder, draft probe,
                expert inference, selection and generation
  E2E           before the raw media is read -> answer text

  python experiments/run_efficiency.py --dataset ws --method omnidraft45
"""
import argparse
import json
import os
import random as _random
import statistics
import time

import eval_kit as kit

NF = {"ws": 128, "daily": 128, "vme": 512}
LIMIT = {"ws": 4000, "daily": 2000, "vme": 3000}
SEED = 20261108
REPS = 3

METHODS = {
    "full": dict(method="full"),
    "dycoke55": dict(method="dycoke", keep_ratio=0.55),
    "omniselect45": dict(method="omniselect", keep_ratio=0.45),
    "omniselect30": dict(method="omniselect", keep_ratio=0.30),
    "omnidraft45": dict(method="omnidraft", keep_ratio=0.45, boost="gap", draft_ratio=0.05,
                        probe_layer=12, lam=0.5, gate=0, exec="rerun"),
    "omnidraft30": dict(method="omnidraft", keep_ratio=0.30, boost="gap", draft_ratio=0.05,
                        probe_layer=12, lam=0.5, gate=0, exec="rerun"),
}
GROUPS = {"vme": ["short", "medium", "long"], "daily": None, "ws": None}


def group_of(ds, sample, meta):
    return meta[sample["key"]]


def build_manifest(ds):
    """Fixed, stratified 30 questions plus 3 warmup ones, all distinct videos."""
    samples = kit.rb.load_samples(ds, LIMIT[ds], 0)
    meta = {}
    groups = []
    if ds == "vme":
        import pyarrow.parquet as pq
        rows = pq.read_table(os.path.join(os.environ["OSELECT_DATA"], "VideoMME/videomme/"
                             "test-00000-of-00001.parquet")).to_pylist()
        for i, r in enumerate(rows):
            meta[f"{r['videoID']}_{i}"] = r["duration"]
        groups = ["short", "medium", "long"]
    elif ds == "daily":
        rows = json.load(open(os.path.join(os.environ["OSELECT_DATA"], "DailyOmni/qa.json")))
        for i, r in enumerate(rows):
            meta[f"{r['video_id']}_{i}"] = r["Type"]
        groups = ["Context understanding", "Event Sequence", "AV Event Alignment",
                  "Comparative", "Inference", "Reasoning"]
    else:
        qa = json.load(open(os.path.join(os.environ["OSELECT_DATA"],
                                         "WorldSense/worldsense_qa.json")))
        for v in qa.values():
            for k, t in v.items():
                if str(k).startswith("task"):
                    meta[f"{v['video_id']}_{k}"] = t.get("domain", "")
        groups = ["Tech & Science", "Culture & Politics", "Daily Life", "Film & TV",
                  "Performance", "Games", "Sports", "Music"]

    by_group = {g: [] for g in groups}
    for s in samples:
        g = meta.get(s["key"])
        if g in by_group:
            by_group[g].append(s)

    rnd = _random.Random(SEED)
    for g in groups:
        rnd.shuffle(by_group[g])

    picked, used_videos = [], set()
    quota = 10 if ds == "vme" else (5 if ds == "daily" else None)
    if quota:
        for g in groups:
            n = 0
            for s in by_group[g]:
                if n >= quota:
                    break
                if s["video"] in used_videos:
                    continue
                picked.append(s)
                used_videos.add(s["video"])
                n += 1
    else:
        # 30 over 8 domains: round robin so every domain appears.
        cursors = {g: 0 for g in groups}
        while len(picked) < 30:
            progressed = False
            for g in groups:
                if len(picked) >= 30:
                    break
                while cursors[g] < len(by_group[g]):
                    s = by_group[g][cursors[g]]
                    cursors[g] += 1
                    if s["video"] in used_videos:
                        continue
                    picked.append(s)
                    used_videos.add(s["video"])
                    progressed = True
                    break
            if not progressed:
                break

    warm, warm_videos = [], set(used_videos)
    for s in samples:
        if len(warm) >= 3:
            break
        if s["video"] in warm_videos:
            continue
        warm.append(s)
        warm_videos.add(s["video"])

    return picked, warm, meta, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ws", "daily", "vme"])
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--model_path", default=os.environ.get("MODEL"))
    ap.add_argument("--out", default="/data/source_new02/olizhan/OmniDraft/results/efficiency_7b")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max_questions", type=int, default=0,
                    help="only the first N questions, for a smoke test")
    ap.add_argument("--force", action="store_true", help="rerun even if a summary exists")
    args = ap.parse_args()

    ds, nf = args.dataset, NF[args.dataset]
    d = os.path.join(args.out, ds)
    os.makedirs(d, exist_ok=True)
    if not args.force and os.path.exists(os.path.join(d, args.method, "summary.json")):
        print(f"skip {ds}/{args.method}: summary.json exists", flush=True)
        return
    mpath = os.path.join(d, "manifest.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        picked, warm = m["questions"], m["warmup"]
        meta, groups = m["meta"], m["groups"]
    else:
        picked, warm, meta, groups = build_manifest(ds)
        with open(mpath, "w") as f:
            json.dump({"seed": SEED, "nframes": nf, "reps": args.reps,
                       "groups": groups, "meta": meta,
                       "questions": picked, "warmup": warm}, f, indent=1)
    if args.max_questions:
        picked, warm = picked[:args.max_questions], warm[:1]
    print(f"{ds}: {len(picked)} questions, {len(warm)} warmup, methods include {args.method}", flush=True)

    md = os.path.join(d, args.method)
    os.makedirs(md, exist_ok=True)
    over = METHODS[args.method]
    with open(os.path.join(md, "config.json"), "w") as f:
        json.dump({"dataset": ds, "method": args.method, "params": over, "nframes": nf,
                   "reps": args.reps, "model": args.model_path, "seed": SEED,
                   "batch_size": 1, "max_new_tokens": 8,
                   "note": "each repeat re-decodes the media; no expert or probe reuse"},
                  f, indent=1, sort_keys=True)

    k = kit.load(args.model_path)

    # Warm up the same code path on questions outside the timed list.
    for s in warm:
        try:
            inp, aud, img, vid, prep_s, short = k.prep(s, nf)
            k.run(k.cfg(**over), inp, vid, aud, s["question"])
            del inp
        except Exception as e:
            print("warmup skip", type(e).__name__, str(e)[:80], flush=True)
        kit.torch.cuda.empty_cache()

    rows = []
    for qi, s in enumerate(picked):
        for rep in range(args.reps):
            rec = {"key": s["key"], "group": meta.get(s["key"]), "repeat": rep,
                   "nframes": nf, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
            try:
                kit.torch.cuda.synchronize()
                peak0 = kit.torch.cuda.max_memory_allocated() / 1024 ** 3
                kit.torch.cuda.reset_peak_memory_stats()
                t_e2e = time.perf_counter()
                inp, aud, img, vid, prep_s, short = k.prep(s, nf)
                if kit.torch.cuda.is_available():
                    kit.torch.cuda.synchronize()
                t_lat = time.perf_counter()
                resp, pred, gen_s, stats, n_out = k.run(
                    k.cfg(**over), inp, vid, aud, s["question"], cache_key=None)
                lat = time.perf_counter() - t_lat
                e2e = time.perf_counter() - t_e2e
                n_a, n_v = k.counts(inp)
                peak = kit.torch.cuda.max_memory_allocated() / 1024 ** 3
                rec.update({
                    "short": short, "prep_s": round(prep_s, 3),
                    "latency_s": round(lat, 3), "e2e_s": round(e2e, 3),
                    "gen_s": round(gen_s, 3), "decode_s": round(max(0.0, lat - gen_s), 3),
                    "prefill_ms": round(k.prefill_ms, 1) if k.prefill_ms else None,
                    "expert_s": round(getattr(k, "last_expert_s", 0.0), 3),
                    "peak_mem_gb": round(peak, 2), "mem_before_gb": round(peak0, 2),
                    "out_tokens": n_out, "n_audio": n_a, "n_video": n_v,
                    "pred": pred, "gt": s["gt"], "is_correct": pred == s["gt"] and pred != "",
                    "ok": True,
                })
                for key in ("draft_audio", "k_audio", "draft_video", "k_video", "probe_sec", "probe_sig_sec", "select_sec"):
                    if key in stats:
                        rec[key] = stats[key]
                del inp
            except kit.torch.cuda.OutOfMemoryError as e:
                kit.torch.cuda.empty_cache()
                rec.update({"ok": False, "oom": True, "err": str(e)[:180]})
            except Exception as e:
                rec.update({"ok": False, "fail": True, "err": str(e)[:180]})
            rows.append(rec)
            with open(os.path.join(md, "measurements.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kit.torch.cuda.empty_cache()
        print(f"{qi+1}/{len(picked)} {s['key']} "
              + " ".join(f"{r['repeat']}:{r.get('e2e_s')}" for r in rows[-args.reps:]), flush=True)

    # Per question median over the successful repeats, then mean over questions.
    per_q = {}
    for r in rows:
        per_q.setdefault(r["key"], []).append(r)
    keys = ["e2e_s", "latency_s", "prefill_ms", "peak_mem_gb", "prep_s", "gen_s", "expert_s", "out_tokens"]
    med = {k2: [] for k2 in keys}
    ok = att = 0
    for q, rs in per_q.items():
        good = [r for r in rs if r.get("ok")]
        att += len(rs)
        ok += len(good)
        for k2 in keys:
            vals = [r[k2] for r in good if isinstance(r.get(k2), (int, float))]
            if vals:
                med[k2].append(statistics.median(vals))
    summary = {
        "questions": len(per_q), "attempted": att, "successful": ok,
        "accuracy_on_subset": round(100.0 * sum(1 for q, rs in per_q.items()
                                                if any(r.get("is_correct") for r in rs)) / max(1, len(per_q)), 1),
        "reps": args.reps, "seed": SEED, "nframes": nf,
    }
    for k2 in keys:
        summary[k2 + "_mean"] = round(sum(med[k2]) / len(med[k2]), 3) if med[k2] else None
    summary["peak_mem_max_gb"] = round(max((r.get("peak_mem_gb") or 0) for r in rows), 2)
    with open(os.path.join(md, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
