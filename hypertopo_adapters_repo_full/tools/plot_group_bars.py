#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grouped bar plots for multiple experiments (Dice, IoU, BF1, Betti errors, PD).
Reads metrics from each run directory:
  - prefer metrics_test.json, else metrics_val.json

Usage:
  python plot_group_bars.py \
    --runs outputs/euclid_only outputs/hes_notopo outputs/hes_topo \
    --labels "E-only" "H⊕E⊕S (no topo)" "H⊕E⊕S + topo" \
    --metrics dice iou boundary_f1 betti0_err betti1_err pd_dist \
    --outfile outputs/summary_grouped_bars.png
"""
import argparse, json, os, sys
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_METRICS = ["dice", "iou", "boundary_f1", "betti0_err", "betti1_err", "pd_dist"]
PRETTY = {
    "dice": "Dice ↑",
    "iou": "IoU ↑",
    "boundary_f1": "Boundary-F1 ↑",
    "betti0_err": "|Δβ₀| ↓",
    "betti1_err": "|Δβ₁| ↓",
    "pd_dist": "PD ↓",
}

def load_metrics(run_dir: str) -> Dict[str, float]:
    cand = [os.path.join(run_dir, "metrics_test.json"),
            os.path.join(run_dir, "metrics_val.json")]
    for p in cand:
        if os.path.isfile(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No metrics_test.json or metrics_val.json in {run_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="List of run directories (each contains metrics_*.json)")
    ap.add_argument("--labels", nargs="+", default=None, help="Display labels (same length as --runs)")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Which metrics to plot")
    ap.add_argument("--outfile", default=None, help="If set, save PNG here; else show interactively")
    args = ap.parse_args()

    runs: List[str] = args.runs
    labels: List[str] = args.labels or [os.path.basename(os.path.normpath(r)) for r in runs]
    if len(labels) != len(runs):
        print("[ERR] --labels length must match --runs length", file=sys.stderr); sys.exit(1)

    metrics_to_plot = args.metrics
    for m in metrics_to_plot:
        if m not in DEFAULT_METRICS:
            print(f"[WARN] Unknown metric '{m}', plotting anyway.")

    # load all metrics
    rows = []
    for r in runs:
        try:
            m = load_metrics(r)
        except Exception as e:
            print(f"[WARN] {r}: {e}")
            m = {k: np.nan for k in DEFAULT_METRICS}
        rows.append([float(m.get(k, np.nan)) for k in metrics_to_plot])

    data = np.array(rows)  # [num_runs, num_metrics]
    num_runs, num_metrics = data.shape

    # plotting
    width = 0.8 / num_runs
    x = np.arange(num_metrics)
    fig, ax = plt.subplots(figsize=(max(6, num_metrics * 1.8), 3.8))

    for i in range(num_runs):
        ax.bar(x + (i - (num_runs-1)/2) * width, data[i], width=width, label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in metrics_to_plot], rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Segmentation summary across runs")
    ax.legend(ncol=min(num_runs, 4), frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        plt.savefig(args.outfile, dpi=200)
        print(f"[OK] saved {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

