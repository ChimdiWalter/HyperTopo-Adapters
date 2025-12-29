#!/usr/bin/env python3
"""
Aggregate training/eval summaries from outputs/.

Reads (if present):
- best_metrics.json
- metrics_val.json
- metrics_test.json
- selection_minPD_in_topK.json

Writes:
- summary_metrics.csv
- summary_metrics.md
"""

import argparse, json, os, glob, sys
from collections import OrderedDict

METRIC_KEYS = [
    "dice", "iou", "boundary_f1",
    "betti0_err", "betti1_err", "pd_dist"
]

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def collect(outputs_dir: str):
    paths = {
        "best_metrics": os.path.join(outputs_dir, "best_metrics.json"),
        "metrics_val": os.path.join(outputs_dir, "metrics_val.json"),
        "metrics_test": os.path.join(outputs_dir, "metrics_test.json"),
        "selection": os.path.join(outputs_dir, "selection_minPD_in_topK.json"),
    }

    data = {k: _load_json(p) if os.path.isfile(p) else None for k, p in paths.items()}
    # Try to detect backbone/geom from any training log dump if available
    meta = {}
    # Optional: sniff a config copy if you store it in outputs
    conf_candidates = glob.glob(os.path.join(outputs_dir, "*config*.json")) + \
                      glob.glob(os.path.join(outputs_dir, "*config*.yml")) + \
                      glob.glob(os.path.join(outputs_dir, "*config*.yaml"))
    meta["config_files_found"] = [os.path.basename(p) for p in conf_candidates]

    return data, meta, paths

def to_row(tag, d):
    row = OrderedDict()
    row["tag"] = tag
    if d is None:
        for k in METRIC_KEYS: row[k] = ""
        return row
    for k in METRIC_KEYS:
        row[k] = _fmt(d.get(k, ""))
    return row

def write_csv_md(outputs_dir: str, rows, topk_info):
    csv_path = os.path.join(outputs_dir, "summary_metrics.csv")
    md_path  = os.path.join(outputs_dir, "summary_metrics.md")

    # CSV
    headers = ["tag"] + METRIC_KEYS
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")

    # Markdown
    with open(md_path, "w") as f:
        f.write("# Summary Metrics\n\n")
        # table
        f.write("| tag | " + " | ".join(METRIC_KEYS) + " |\n")
        f.write("|" + "---|" * (len(METRIC_KEYS) + 1) + "\n")
        for r in rows:
            f.write("| " + r["tag"] + " | " + " | ".join(str(r[k]) for k in METRIC_KEYS) + " |\n")
        f.write("\n")

        # top-k info
        if topk_info:
            f.write("## Top-K (by Dice) and min-PD selection\n\n")
            chosen = topk_info.get("chosen", {})
            tk = topk_info.get("topK", [])
            if tk:
                f.write("**Top-K by Dice**:\n\n")
                f.write("| epoch | dice | pd |\n|---|---:|---:|\n")
                for c in tk:
                    f.write(f"| {c.get('epoch','')} | {_fmt(c.get('dice',''))} | {_fmt(c.get('pd',''))} |\n")
                f.write("\n")
            if chosen:
                f.write(f"**Chosen (min-PD within Top-K)**: epoch={chosen.get('epoch','')}, "
                        f"Dice={_fmt(chosen.get('dice',''))}, PD={_fmt(chosen.get('pd',''))}\n\n")

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="./outputs",
                    help="Path to the outputs directory containing JSONs")
    args = ap.parse_args()

    outputs_dir = os.path.abspath(args.outputs_dir)
    if not os.path.isdir(outputs_dir):
        print(f"[ERR] outputs_dir not found: {outputs_dir}")
        sys.exit(1)

    data, meta, paths = collect(outputs_dir)

    rows = []
    rows.append(to_row("metrics_val",  data.get("metrics_val")))
    rows.append(to_row("metrics_test", data.get("metrics_test")))
    rows.append(to_row("best_metrics", data.get("best_metrics")))

    # selection info (Top-K by Dice + min-PD among them)
    topk_info = data.get("selection") or {}

    write_csv_md(outputs_dir, rows, topk_info)

    # also echo a concise console summary
    print("\n[SUMMARY]")
    for r in rows:
        print(" -", r["tag"], {k: r[k] for k in METRIC_KEYS})
    if topk_info:
        chosen = topk_info.get("chosen", {})
        if chosen:
            print(f" - chosen(min-PD in Top-K): epoch={chosen.get('epoch')}, "
                  f"Dice={_fmt(chosen.get('dice'))}, PD={_fmt(chosen.get('pd'))}")
    if meta["config_files_found"]:
        print(" - config files found in outputs:", meta["config_files_found"])

if __name__ == "__main__":
    main()

