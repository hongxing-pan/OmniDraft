"""Sensitivity sweeps for OmniDraft_实验表格.tex (Qwen2.5-Omni-7B, 30% kept).

Eleven distinct configs, because the main setting (5% draft, depth 12, lam 0.5)
sits in all three sweeps:

  draft  5 / 10 / 15 / 20 %   at depth 12, lam 0.5
  depth  4 / 8 / 12 / 16      at draft 5 %, lam 0.5
  lam    0 / .25 / .5 / .75 / 1  at draft 5 %, depth 12

WorldSense (3,172 q) and DailyOmni (1,197 q), 128 frames, keep 0.30. The media is
decoded once per question and every config reuses those tensors, which is what
makes 11 configs affordable; `e2e` is then reported as the measured prep time of
that question plus the config's own latency, so the decode cost is still counted
once per config. Accuracy is unaffected by the sharing.

  python experiments/run_sensitivity.py --dataset ws --shard 0 --nshards 8
"""
import argparse
import json
import os
import time

import eval_kit as kit


def configs():
    """(config_id, overrides) in sweep order, deduplicated."""
    out = []

    def add(cid, **over):
        for c, _ in out:
            if c == cid:
                return
        out.append((cid, over))

    base = dict(method="omnidraft", keep_ratio=0.30, boost="gap",
                gate=0, exec="rerun")
    for d in (5, 10, 15, 20):
        add(f"draft{d:02d}_p12_l050", draft_ratio=d / 100.0, probe_layer=12, lam=0.50, **base)
    for p in (4, 8, 12, 16):
        add(f"draft05_p{p:02d}_l050", draft_ratio=0.05, probe_layer=p, lam=0.50, **base)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        add(f"draft05_p12_l{int(round(lam * 100)):03d}", draft_ratio=0.05,
            probe_layer=12, lam=lam, **base)
    return out


NF = {"ws": 128, "daily": 128}
LIMIT = {"ws": 4000, "daily": 2000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ws", "daily"])
    ap.add_argument("--model_path", default=os.environ.get("MODEL"))
    ap.add_argument("--out", default="/data/source_new02/olizhan/OmniDraft/results/sensitivity")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    ds, nf = args.dataset, NF[args.dataset]
    samples = kit.rb.load_samples(ds, LIMIT[ds], 0)
    mine = [s for i, s in enumerate(samples) if i % args.nshards == args.shard]

    confs = configs()
    dirs = {}
    for cid, over in confs:
        d = os.path.join(args.out, ds, cid)
        os.makedirs(d, exist_ok=True)
        dirs[cid] = d
        cfg = dict(kit.BASE_CFG)
        cfg.update(over)
        cfg.update({"model": args.model_path, "dataset": ds, "nframes": nf,
                    "keep_ratio": 0.30, "image_pixels": "nframes*128*28*28",
                    "limit": LIMIT[ds], "offset": 0, "shard": args.shard,
                    "nshards": args.nshards,
                    "note": "e2e = measured prep of this question + this config's latency"})
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1, sort_keys=True)

    k = kit.load(args.model_path)

    # One unrecorded question so the first config does not carry CUDA/kernel warmup.
    if mine:
        w = samples[(-1 - args.shard) % len(samples)]
        try:
            inp, aud, img, vid, _, _ = k.prep(w, nf)
            for cid, over in confs:
                k.run(k.cfg(**over), inp, vid, aud, w["question"])
            del inp
            kit.torch.cuda.empty_cache()
        except Exception as e:
            print("warmup skip", type(e).__name__, str(e)[:80], flush=True)
    print(f"{ds}: {len(mine)} questions x {len(confs)} configs", flush=True)

    sums = {cid: {"n": 0, "ok": 0, "fail": 0, "probe": 0.0, "e2e": 0.0, "lat": 0.0}
            for cid, _ in confs}
    t_start = time.perf_counter()
    for i, s in enumerate(mine):
        try:
            inputs, audios, images, videos, prep_s, short = k.prep(s, nf)
        except Exception as e:
            for cid, _ in confs:
                with open(os.path.join(dirs[cid], "predictions.jsonl"), "a") as f:
                    f.write(json.dumps({"key": s["key"], "fail": True,
                                        "err": str(e)[:180]}, ensure_ascii=False) + "\n")
                sums[cid]["fail"] += 1
            print(f"{i} PREP FAIL {type(e).__name__}: {e}"[:160], flush=True)
            continue
        n_a, n_v = k.counts(inputs)

        for cid, over in confs:
            cfg = k.cfg(**over)
            rec = {"key": s["key"], "short": short, "n_audio": n_a, "n_video": n_v}
            try:
                if kit.torch.cuda.is_available():
                    kit.torch.cuda.synchronize()
                t0 = time.perf_counter()
                resp, pred, gen_s, stats, n_out = k.run(cfg, inputs, videos, audios, s["question"])
                lat = time.perf_counter() - t0
                ok = pred == s["gt"] and pred != ""
                rec.update({
                    "prediction": resp, "pred": pred, "gt": s["gt"], "is_correct": ok,
                    "gen_s": round(gen_s, 3), "latency_s": round(lat, 3),
                    "prep_s": round(prep_s, 3), "e2e_s": round(prep_s + lat, 3),
                    "probe_ms": round(float(stats.get("probe_sig_sec", 0.0)) * 1000.0, 1),
                    "probe_only_ms": round(float(stats.get("probe_sec", 0.0)) * 1000.0, 1),
                    "select_ms": round(float(stats.get("select_sec", 0.0)) * 1000.0, 1),
                    "draft_audio": stats.get("draft_audio"), "keep_audio": stats.get("k_audio"),
                    "draft_video": stats.get("draft_video"), "keep_video": stats.get("k_video"),
                    "lam_used": stats.get("lam"), "g": stats.get("g"), "out_tokens": n_out,
                })
                st = sums[cid]
                st["n"] += 1
                st["ok"] += int(ok)
                st["probe"] += rec["probe_ms"]
                st["e2e"] += rec["e2e_s"]
                st["lat"] += rec["latency_s"]
            except kit.torch.cuda.OutOfMemoryError as e:
                kit.torch.cuda.empty_cache()
                rec.update({"oom": True, "err": str(e)[:180]})
                sums[cid]["fail"] += 1
            except Exception as e:
                rec.update({"fail": True, "err": str(e)[:180]})
                sums[cid]["fail"] += 1
            with open(os.path.join(dirs[cid], "predictions.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        del inputs
        kit.torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t_start
            print(f"[progress] {i+1}/{len(mine)} questions, {el/(i+1):.1f}s each, "
                  f"elapsed {el/60:.0f} min", flush=True)

    for cid, _ in confs:
        st = sums[cid]
        n = st["n"]
        summary = {
            "completed": n, "correct": st["ok"], "fail_or_oom": st["fail"],
            "accuracy": round(100.0 * st["ok"] / n, 2) if n else None,
            "probe_ms_mean": round(st["probe"] / n, 1) if n else None,
            "latency_s_mean": round(st["lat"] / n, 3) if n else None,
            "e2e_s_mean": round(st["e2e"] / n, 3) if n else None,
            "shard": args.shard, "nshards": args.nshards,
        }
        with open(os.path.join(dirs[cid], f"summary_shard{args.shard}.json"), "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
        print(cid, summary, flush=True)


if __name__ == "__main__":
    main()
