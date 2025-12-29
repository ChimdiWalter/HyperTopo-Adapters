#!/usr/bin/env python3
import argparse, os, json
from collections import OrderedDict

METRIC_KEYS = ["dice","iou","boundary_f1","betti0_err","betti1_err","pd_dist"]

def _load(path):
    try:
        with open(path,"r") as f: return json.load(f)
    except: return None

def row(tag, d):
    r = OrderedDict(tag=tag)
    for k in METRIC_KEYS:
        v = d.get(k,"") if d else ""
        try: v = f"{float(v):.4f}"
        except: pass
        r[k]=v
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="One or more experiment output dirs")
    ap.add_argument("--out", default="./outputs/summary_all.csv")
    ap.add_argument("--which", default="val,test,best", help="comma list among val,test,best")
    args = ap.parse_args()
    which = [w.strip() for w in args.which.split(",") if w.strip()]

    rows = []
    for root in args.roots:
        tag = os.path.basename(os.path.normpath(root))
        mv = _load(os.path.join(root,"metrics_val.json"))
        mt = _load(os.path.join(root,"metrics_test.json"))
        bm = _load(os.path.join(root,"best_metrics.json"))
        if "val" in which:  rows.append(row(f"{tag}:val",  mv))
        if "test" in which: rows.append(row(f"{tag}:test", mt))
        if "best" in which: rows.append(row(f"{tag}:best", bm))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # CSV
    with open(args.out,"w") as f:
        f.write("tag," + ",".join(METRIC_KEYS) + "\n")
        for r in rows:
            f.write(",".join([r["tag"]] + [str(r[k]) for k in METRIC_KEYS]) + "\n")

    # MD sidecar
    md = args.out.rsplit(".",1)[0] + ".md"
    with open(md,"w") as f:
        f.write("# Summary (multi-root)\n\n")
        f.write("| tag | " + " | ".join(METRIC_KEYS) + " |\n")
        f.write("|---|" + "|".join(["---:"]*len(METRIC_KEYS)) + "|\n")
        for r in rows:
            f.write("| " + r["tag"] + " | " + " | ".join(str(r[k]) for k in METRIC_KEYS) + " |\n")
    print(f"[OK] wrote {args.out}\n[OK] wrote {md}")

if __name__ == "__main__":
    main()
