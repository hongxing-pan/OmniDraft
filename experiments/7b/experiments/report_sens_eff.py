"""Merge the sensitivity/efficiency shards and fill the tables.

Accuracy in the efficiency table is taken from the full benchmark runs under the
same input settings (results/7b/preds_*.csv), never from the 30 timing questions.
DyCoke was timed at 55% but the main table was run at 50%, so its accuracy cell
is left empty on purpose.

  python experiments/report_sens_eff.py [--tex out.tex]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

SENS = "/data/source_new02/olizhan/OmniDraft/results/sensitivity"
EFF = "/data/source_new02/olizhan/OmniDraft/results/efficiency_7b"
PREDS = "/data/source_new02/olizhan/OmniDraft/results/7b"
DATASETS = ["ws", "daily", "vme"]
DOC_NAME = {"ws": "WorldSense", "daily": "DailyOmni", "vme": "VideoMME"}

DRAFT_SWEEP = [(5, "draft05_p12_l050"), (10, "draft10_p12_l050"),
               (15, "draft15_p12_l050"), (20, "draft20_p12_l050")]
DEPTH_SWEEP = [(4, "draft05_p04_l050"), (8, "draft05_p08_l050"),
               (12, "draft05_p12_l050"), (16, "draft05_p16_l050")]
LAM_SWEEP = [(0.0, "draft05_p12_l000"), (0.25, "draft05_p12_l025"),
             (0.5, "draft05_p12_l050"), (0.75, "draft05_p12_l075"),
             (1.0, "draft05_p12_l100")]

EFF_METHODS = [
    ("full", "Full Tokens"),
    ("dycoke55", "DyCoke (V\\&A, 55\\%)"),
    ("omniselect45", "OmniSelect (45\\%)"),
    ("omniselect30", "OmniSelect (30\\%)"),
    ("omnidraft45", "\\textbf{OmniDraft (45\\%)}"),
    ("omnidraft30", "\\textbf{OmniDraft (30\\%)}"),
]
ACC_ROW = {"full": "full", "dycoke55": None, "omniselect45": "omniselect45",
           "omniselect30": "omniselect30", "omnidraft45": "omnidraft45",
           "omnidraft30": "omnidraft30"}


def sens_merge(ds):
    """config_id -> weighted summary over shards."""
    out = {}
    for d in glob.glob(os.path.join(SENS, ds, "*")):
        cid = os.path.basename(d)
        tot = {"n": 0, "ok": 0, "bad": 0, "probe": 0.0, "e2e": 0.0}
        found = False
        for f in glob.glob(os.path.join(d, "summary_shard*.json")):
            s = json.load(open(f))
            found = True
            n = s.get("completed") or 0
            tot["n"] += n
            tot["ok"] += s.get("correct") or 0
            tot["bad"] += s.get("fail_or_oom") or 0
            tot["probe"] += (s.get("probe_ms_mean") or 0.0) * n
            tot["e2e"] += (s.get("e2e_s_mean") or 0.0) * n
        if not found or not tot["n"]:
            continue
        out[cid] = {
            "n": tot["n"], "ok": tot["ok"], "bad": tot["bad"],
            "acc": 100.0 * tot["ok"] / tot["n"],
            "probe_ms": tot["probe"] / tot["n"],
            "e2e_s": tot["e2e"] / tot["n"],
        }
    return out


def eff_read(ds, method):
    p = os.path.join(EFF, ds, method, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def acc_lookup():
    """full-run accuracy per (row, dataset) from the exported per-question csv."""
    out = {}
    for ds in DATASETS:
        p = os.path.join(PREDS, f"preds_{ds}.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            if not r.get("full_ok"):
                continue
            for col in r:
                if not col.endswith("_ok"):
                    continue
                row = col[:-3]
                st = out.setdefault((row, ds), [0, 0])
                if r[col] != "":
                    st[1] += 1
                    st[0] += int(r[col])
    return {k: 100.0 * v[0] / v[1] for k, v in out.items() if v[1]}


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="/data/source_new02/olizhan/OmniDraft/results/omnidraft_tables_filled.tex")
    args = ap.parse_args()

    sens = {ds: sens_merge(ds) for ds in ("ws", "daily")}
    acc = acc_lookup()

    lines = []
    for ds in ("ws", "daily"):
        print(f"== sensitivity {doc_name(ds)} ==")
        for label, sweep in (("draft", DRAFT_SWEEP), ("depth", DEPTH_SWEEP), ("lambda", LAM_SWEEP)):
            for key, cid in sweep:
                s = sens[ds].get(cid)
                if not s:
                    continue
                print(f"  {label} {key}: acc {s['acc']:.1f} ({s['ok']}/{s['n']}, "
                      f"bad {s['bad']}) probe {s['probe_ms']:.0f}ms e2e {s['e2e_s']:.2f}s")
        print()

    print("== efficiency ==")
    for ds in DATASETS:
        for m, _ in EFF_METHODS:
            s = eff_read(ds, m)
            print(f"  {doc_name(ds):10s} {m:13s} "
                  + ("missing" if not s else
                     f"mem {fmt(s.get('peak_mem_gb_mean'),2)}GB prefill {fmt(s.get('prefill_ms_mean'))}ms "
                     f"lat {fmt(s.get('latency_s_mean'),3)}s e2e {fmt(s.get('e2e_s_mean'),3)}s "
                     f"ok {s['successful']}/{s['attempted']}"))
        print()

    # ---- tex ----
    def cell(v, nd=1):
        return "" if v is None else f"{v:.{nd}f}"

    tex = []
    tex.append("% Generated by experiments/report_sens_eff.py; numbers come from")
    tex.append("% results/omnidraft/sensitivity and results/omnidraft/efficiency_7b.")
    tex.append("")

    def sweep_table(caption, label, sweep, keyfmt, head):
        out = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}",
               r"\label{" + label + "}", r"\setlength{\tabcolsep}{8pt}",
               r"\renewcommand{\arraystretch}{1.08}", r"\resizebox{\textwidth}{!}{",
               r"\begin{tabular}{@{}c*{6}{c}@{}}", r"\toprule",
               r"\multirow{2}{*}{" + head + "}",
               r"& \multicolumn{3}{c}{WorldSense} & \multicolumn{3}{c}{DailyOmni} \\",
               r"\cmidrule(lr){2-4}", r"\cmidrule(lr){5-7}",
               r"& Acc. (\%) $\uparrow$ & Probe (ms) $\downarrow$ & E2E (s) $\downarrow$"
               r" & Acc. (\%) $\uparrow$ & Probe (ms) $\downarrow$ & E2E (s) $\downarrow$ \\",
               r"\midrule"]
        for key, cid in sweep:
            w = sens["ws"].get(cid)
            d = sens["daily"].get(cid)
            out.append(f"{keyfmt(key)} & "
                       + " & ".join([
                           cell(w["acc"]) if w else "", cell(w["probe_ms"], 0) if w else "", cell(w["e2e_s"], 2) if w else "",
                           cell(d["acc"]) if d else "", cell(d["probe_ms"], 0) if d else "", cell(d["e2e_s"], 2) if d else "",
                       ]) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
        return out

    tex += sweep_table(
        "Sensitivity to the draft token percentage on Qwen2.5-Omni-7B with 30\\% final token "
        "retention and 128 input frames. Draft percentages are relative to the original token "
        "count of each modality. Other settings follow the main experiments.",
        "tab:sensitivity_draft", DRAFT_SWEEP, lambda k: str(k), r"Draft Tokens (\%)")
    tex += sweep_table(
        "Sensitivity to the probe depth on Qwen2.5-Omni-7B with 30\\% final token retention and "
        "128 input frames. Depth counts the decoder blocks the probe has passed. Other settings "
        "follow the main experiments.",
        "tab:sensitivity_depth", DEPTH_SWEEP, lambda k: str(k), "Probe Depth")

    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Sensitivity to the demand mixing coefficient on Qwen2.5-Omni-7B with 30\% "
           r"final token retention and 128 input frames. Other settings follow the main "
           r"experiments. $\lambda=0$ needs no probe, so its probe time is zero.}",
           r"\label{tab:sensitivity_lambda}", r"\small", r"\setlength{\tabcolsep}{10pt}",
           r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}ccc@{}}", r"\toprule",
           r"$\lambda$ & WorldSense $\uparrow$ & DailyOmni $\uparrow$ \\", r"\midrule"]
    for key, cid in LAM_SWEEP:
        w = sens["ws"].get(cid)
        d = sens["daily"].get(cid)
        out.append(f"{key:.2f} & {cell(w['acc']) if w else ''} & {cell(d['acc']) if d else ''} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex += out

    tex += [r"\begin{table*}[t]", r"\centering",
            r"\caption{Efficiency comparison on Qwen2.5-Omni-7B. Timing and memory are measured "
            r"on 30 fixed questions per benchmark. Accuracy is taken from the full benchmark "
            r"evaluation under the same input settings; DyCoke is left blank because the full "
            r"run used 50\% while this timing run uses 55\%. Latency excludes common media "
            r"loading and preprocessing; E2E includes them. Both include all method-specific "
            r"overhead, including OmniSelect's expert inference and OmniDraft's draft probing. "
            r"LLM prefill measures the final answer pass only.}",
            r"\label{tab:efficiency}", r"\setlength{\tabcolsep}{6pt}",
            r"\renewcommand{\arraystretch}{1.08}", r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"Method & \makecell{GPU Mem.\\(GB) $\downarrow$} & \makecell{LLM Prefill\\(ms) $\downarrow$}"
            r" & \makecell{Acc.\\(\%) $\uparrow$} & \makecell{Latency\\(s) $\downarrow$}"
            r" & \makecell{E2E Time\\(s) $\downarrow$} \\", r"\midrule"]
    for i, ds in enumerate(DATASETS):
        if i:
            tex.append(r"\midrule")
        tex.append(r"\multicolumn{6}{c}{\textit{" + DOC_NAME[ds] + "}} " + r"\\")
        tex.append(r"\midrule")
        for m, label in EFF_METHODS:
            s = eff_read(ds, m)
            a = ACC_ROW[m]
            a = acc.get((a, ds)) if a else None
            if not s:
                tex.append(label + r" & & & " + (cell(a) if a else "") + r" & & \\")
                continue
            tex.append(label + " & " + " & ".join([
                cell(s.get("peak_mem_gb_mean"), 2),
                cell(s.get("prefill_ms_mean"), 0),
                cell(a),
                cell(s.get("latency_s_mean"), 3),
                cell(s.get("e2e_s_mean"), 3),
            ]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]

    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    open(args.tex, "w").write("\n".join(tex))
    print("wrote", args.tex)

    md = ["# Sensitivity and efficiency, Qwen2.5-Omni-7B", "",
          "Sensitivity: 30% kept, 128 frames, full splits. Probe time starts at the draft",
          "input and ends when the shared demand signal is ready; E2E adds media reading.", "",
          "## Sensitivity: draft token percentage", "",
          "| Draft % | WorldSense Acc. | Probe | E2E | DailyOmni Acc. | Probe | E2E |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for key, cid in DRAFT_SWEEP:
        w, d = sens["ws"].get(cid), sens["daily"].get(cid)
        md.append(f"| {key} | " + (f"{w['acc']:.1f} ({w['ok']}/{w['n']}) | {w['probe_ms']:.0f} ms | {w['e2e_s']:.2f} s | " if w else "| | | ")
                  + (f"{d['acc']:.1f} ({d['ok']}/{d['n']}) | {d['probe_ms']:.0f} ms | {d['e2e_s']:.2f} s |" if d else "| | |"))
    md += ["", "## Sensitivity: probe depth", "",
           "| Depth | WorldSense Acc. | Probe | E2E | DailyOmni Acc. | Probe | E2E |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, cid in DEPTH_SWEEP:
        w, d = sens["ws"].get(cid), sens["daily"].get(cid)
        md.append(f"| {key} | " + (f"{w['acc']:.1f} ({w['ok']}/{w['n']}) | {w['probe_ms']:.0f} ms | {w['e2e_s']:.2f} s | " if w else "| | | ")
                  + (f"{d['acc']:.1f} ({d['ok']}/{d['n']}) | {d['probe_ms']:.0f} ms | {d['e2e_s']:.2f} s |" if d else "| | |"))
    md += ["", "## Sensitivity: demand mixing coefficient", "",
           "| lambda | WorldSense Acc. | DailyOmni Acc. |", "|---|---:|---:|"]
    for key, cid in LAM_SWEEP:
        w, d = sens["ws"].get(cid), sens["daily"].get(cid)
        md.append(f"| {key:.2f} | " + (f"{w['acc']:.1f} ({w['ok']}/{w['n']})" if w else "")
                  + " | " + (f"{d['acc']:.1f} ({d['ok']}/{d['n']})" if d else "") + " |")
    md += ["", "## Efficiency (30 fixed questions, 3 repeats, batch 1)", "",
           "Acc. is the full-split main-table number; DyCoke is blank because the timing run",
           "uses 55% while the main table uses 50%.", "",
           "| Benchmark | Method | GPU Mem. (GB) | LLM Prefill (ms) | Acc. | Latency (s) | E2E (s) | ok/attempted |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for m, label in EFF_METHODS:
            s = eff_read(ds, m)
            if not s:
                continue
            a = ACC_ROW[m]
            a = acc.get((a, ds)) if a else None
            plain = label.replace(r"\textbf{", "").replace("}", "").replace(r"\&", "&").replace(r"\%", "%")
            md.append(f"| {DOC_NAME[ds]} | {plain} | {s.get('peak_mem_gb_mean'):.2f} | "
                      f"{s.get('prefill_ms_mean'):.0f} | {'' if a is None else f'{a:.1f}'} | "
                      f"{s.get('latency_s_mean'):.3f} | {s.get('e2e_s_mean'):.3f} | "
                      f"{s['successful']}/{s['attempted']} |")
    mdp = os.path.join(os.path.dirname(args.tex), "omnidraft_sens_eff_summary.md")
    open(mdp, "w").write("\n".join(md) + "\n")
    print("wrote", mdp)


def doc_name(ds):
    return DOC_NAME[ds]


if __name__ == "__main__":
    main()
