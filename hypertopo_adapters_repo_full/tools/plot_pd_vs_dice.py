#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter of PD distance vs Dice using selection_minPD_in_topK.json per run.
Also overlays best-by-Dice (if best_metrics.json exists) and the chosen point.

Usage:
  python plot_pd_vs_dice.py \
    --runs outputs/euclid_only outputs/hes_topo \
    --labels "E-only" "H⊕E⊕S+topo" \
    --outfile outputs/pd_vs_dice.png
"""
import argparse, os, json
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

def load_selection(path: str):
    sel_path = os.path.join(path, "selection_minPD_in_topK.json")
    pts = []
    chosen = None
    if os.path.isfile(sel_path):
        with open(sel_path, "r") as f:
            sel = json.load(f)
        for r in sel.get("topK", []):
            pts.append((float(r["dice"]), float(r["pd"])))
        ch = sel.get("chosen", None)
        if ch:
            chosen = (float(ch["dice"]), float(ch["pd"]))
    # best-by-dice reference
    best_path = os.path.join(path, "best_metrics.json")
    best = None
    if os.path.isfile(best_path):
        with open(best_path, "r") as f:
            bm = json.load(f)
        best = (float(bm.get("dice", np.nan)), float(bm.get("pd_dist", np.nan)))
    return pts, chosen, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    runs: List[str] = args.runs
    labels: List[str] = args.labels or [os.path.basename(os.path.normpath(r)) for r in runs]
    assert len(labels) == len(runs), "--labels must match --runs length"

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for r, lbl in zip(runs, labels):
        pts, chosen, best = load_selection(r)
        if len(pts):
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            ax.scatter(x, y, label=f"{lbl} top-K", alpha=0.75, marker="o", s=36)
        if best is not None and all(np.isfinite(best)):
            ax.scatter([best[0]], [best[1]], marker="^", s=70, edgecolor="k",
                       label=f"{lbl} best-by-Dice", zorder=5)
        if chosen is not None and all(np.isfinite(chosen)):
            ax.scatter([chosen[0]], [chosen[1]], marker="*", s=120, edgecolor="k",
                       label=f"{lbl} min-PD in top-K", zorder=6)

    ax.set_xlabel("Dice (val)")
    ax.set_ylabel("PD distance (val)")
    ax.set_title("PD vs Dice (selection view)")
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        plt.savefig(args.outfile, dpi=200)
        print(f"[OK] saved {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

