#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative exports: image / ground-truth / prediction triptychs, with
threshold analysis to explain "looks good but numbers are bad".

Usage:
  python qualitative.py \
    --config configs/default.yaml \
    --ckpt outputs/hes_topo/ckpt_minPD_in_topK.pt \
    --split val \
    --num 12 \
    --outfile outputs/qual_val_grid.png \
    --sweep \
    --calib_subset 64 \
    --sort_by_gain \
    --thresh 0.5

Notes:
- --sweep computes per-image best threshold from a small grid (e.g., 0.2..0.8).
- --calib_subset computes a dataset-best threshold once (over first N samples).
- --sort_by_gain orders rows by how much Dice improves when we re-threshold.
"""

import argparse, os, json, random
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader

from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import make_tokenizer, SpatialDecoder, SimpleDecoder
from src.utils.metrics import dice_coeff, iou, boundary_f1
from src.utils.metrics import betti_numbers as _betti_numbers  # numpy impl

# -------------------- helpers --------------------
def choose_ckpt(out_dir: str, arg_ckpt: str | None) -> str | None:
    if arg_ckpt and os.path.isfile(arg_ckpt):
        return arg_ckpt
    p = os.path.join(out_dir, "ckpt_minPD_in_topK.pt")
    if os.path.isfile(p): return p
    p = os.path.join(out_dir, "best_adapter.pt")
    if os.path.isfile(p): return p
    return None

@torch.no_grad()
def run_model(model, x: torch.Tensor) -> torch.Tensor:
    tokenizer, head, decoder = model["tokenizer"], model["head"], model["decoder"]
    tokens, _ = tokenizer(x)
    lat = head(tokens)
    pooled = lat['tangent'].mean(dim=1)
    pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
    pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))
    return pred  # [B,1,H,W] in [0,1]

def tensor_to_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0,1).numpy()
    if x.shape[0] == 1: x = np.repeat(x, 3, axis=0)
    return (x.transpose(1,2,0) * 255.0).round().astype(np.uint8)

def mask_to_rgb(m: torch.Tensor, color=(255,0,0), alpha=0.4) -> np.ndarray:
    m = (m.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    H, W = m.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[m == 1] = np.array(color, dtype=np.uint8)
    return overlay, alpha

def draw_triptych(ax, img, gt, pred, gt_color=(80,220,120), pr_color=(255,60,60)):
    # left: image, mid: GT overlay, right: pred overlay
    panels = []
    panels.append(img)

    gt_rgb, a = mask_to_rgb(gt, color=gt_color, alpha=0.35)
    gt_vis = (img * (1 - a) + gt_rgb * a).astype(np.uint8)
    panels.append(gt_vis)

    pr_rgb, a = mask_to_rgb(pred, color=pr_color, alpha=0.35)
    pr_vis = (img * (1 - a) + pr_rgb * a).astype(np.uint8)
    panels.append(pr_vis)

    for j, p in enumerate(panels):
        ax[j].imshow(p); ax[j].axis("off")

def hard_metrics(prob: torch.Tensor, gt: torch.Tensor, thr: float) -> Tuple[float,float,float]:
    p = (prob > thr).float()
    d = float(dice_coeff(p, gt).item())
    j = float(iou(p, gt).item())
    bf= float(boundary_f1(p, gt).item())
    return d, j, bf

def betti0_count(mask: torch.Tensor, thr: float) -> int:
    m = (mask.detach().cpu().squeeze().numpy() > thr).astype(np.uint8)
    b0, _ = _betti_numbers(m)
    return int(b0)

def best_threshold_for_image(prob: torch.Tensor, gt: torch.Tensor, grid=None) -> Tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.2, 0.8, 13)  # 0.2,0.25,...,0.8
    best_d, best_t = -1, 0.5
    for t in grid:
        d, _, _ = hard_metrics(prob, gt, t)
        if d > best_d:
            best_d, best_t = d, float(t)
    return best_t, best_d

def dataset_best_threshold(model, ds, device, K=64, grid=None) -> float:
    """Estimate a single dataset threshold from first K samples."""
    if grid is None: grid = np.linspace(0.2, 0.8, 13)
    idxs = list(range(min(len(ds), K)))
    scores = np.zeros_like(grid, dtype=float)
    cnt = 0
    for i in idxs:
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        y = s["mask"].unsqueeze(0).to(device)
        p = run_model(model, x)
        for gi, t in enumerate(grid):
            d, _, _ = hard_metrics(p, y, float(t))
            scores[gi] += d
        cnt += 1
    scores /= max(cnt, 1)
    return float(grid[int(scores.argmax())])

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["val","test","train"], default="val")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--use_simple_decoder", action="store_true")
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--thresh", type=float, default=0.5, help="fixed display threshold")
    ap.add_argument("--sweep", action="store_true", help="show per-image best threshold metrics")
    ap.add_argument("--calib_subset", type=int, default=0, help="compute dataset-best threshold over first N samples")
    ap.add_argument("--sort_by_gain", action="store_true", help="sort rows by Dice gain from re-thresholding")
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)

    # dataset
    dcfg = cfg['data']
    patch_hw = tuple(dcfg['patch_size'])
    csv_map = {"train": dcfg["train_csv"], "val": dcfg["val_csv"], "test": dcfg["test_csv"]}
    ds = LesionDataset(csv_map[args.split], dcfg['image_key'], dcfg['mask_key'], patch_hw)

    # sample indices
    idxs = list(range(len(ds)))
    if args.shuffle:
        random.shuffle(idxs)
    idxs = idxs[:args.num]

    # model
    mcfg = cfg['model']
    token_dim     = int(mcfg.get('token_dim', 256))
    backbone_name = mcfg.get('backbone', 'tiny_tokenizer')
    eu_only       = bool(mcfg.get('euclidean_only', False))
    train_tok     = mcfg.get('train_tokenizer', False)
    unfreeze_keys = mcfg.get('unfreeze_keys', [])

    tokenizer = make_tokenizer(backbone_name, token_dim, train_tok, unfreeze_keys).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1,3,*patch_hw, device=device)
        _, out_hw = tokenizer(dummy)

    head = ProductManifoldHead(
        in_dim=token_dim,
        dh=mcfg['latent_dims']['dh'],
        de=mcfg['latent_dims']['de'],
        ds=mcfg['latent_dims']['ds'],
        learn_curvature=mcfg['curvature']['learn'],
        euclidean_only=eu_only
    ).to(device)

    lat = mcfg['latent_dims']
    tangent_dim = lat['de'] if eu_only else (lat['de'] + lat['dh'] + lat['ds'])
    dec_in_dim = token_dim + tangent_dim
    decoder = (SimpleDecoder(in_dim=dec_in_dim, out_hw=out_hw).to(device)
               if args.use_simple_decoder else
               SpatialDecoder(in_dim=dec_in_dim, low_hw=out_hw).to(device))

    # checkpoint
    out_dir = cfg['logging']['out_dir']
    ckpt_path = choose_ckpt(out_dir, args.ckpt)
    if ckpt_path:
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "head" in ckpt:      head.load_state_dict(ckpt["head"], strict=False)
        if "decoder" in ckpt:   decoder.load_state_dict(ckpt["decoder"], strict=False)
        if "tokenizer" in ckpt: tokenizer.load_state_dict(ckpt["tokenizer"], strict=False)
    else:
        print("[WARN] No checkpoint found; using random head/decoder weights.")

    model = {"tokenizer": tokenizer.eval(), "head": head.eval(), "decoder": decoder.eval()}

    # optional: dataset-wide threshold calibration
    dataset_thr = None
    if args.calib_subset > 0:
        dataset_thr = dataset_best_threshold(model, ds, device, K=args.calib_subset)
        print(f"[CALIB] dataset-best threshold over first {args.calib_subset} = {dataset_thr:.2f}")

    # render grid
    rows = len(idxs)
    cols = 3
    fig_h = max(4, int(1.8 * rows))
    fig_w = 10
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.array([axes])  # normalize

    # optionally sort by “gain from rethresholding”
    order = []
    per_row_titles = []

    for r, i in enumerate(idxs):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        y = s["mask"].unsqueeze(0).to(device)
        p = run_model(model, x)  # [1,1,H,W]

        # fixed-threshold metrics
        d0, j0, bf0 = hard_metrics(p, y, args.thresh)

        # optional per-image best threshold
        best_t, best_d = (args.thresh, d0)
        if args.sweep:
            best_t, best_d = best_threshold_for_image(p, y)

        # optional dataset-best metrics
        d_ds, j_ds, bf_ds = (np.nan, np.nan, np.nan)
        if dataset_thr is not None:
            d_ds, j_ds, bf_ds = hard_metrics(p, y, dataset_thr)

        # counts (Betti0) at 0.5 for explainability
        b0_gt = betti0_count(y, 0.5)
        b0_pr = betti0_count(p, 0.5)

        # Prepare titles
        title_fixed = f"T={args.thresh:.2f} | D {d0:.3f} J {j0:.3f} BF1 {bf0:.3f}"
        title_best  = f"T*={best_t:.2f} D* {best_d:.3f}" if args.sweep else " "
        title_ds    = (f"Tds={dataset_thr:.2f} Dds {d_ds:.3f}" if dataset_thr is not None else " ")
        row_title   = f"{title_fixed}   {title_best}   {title_ds}   | β0 GT={b0_gt} PR={b0_pr}"
        per_row_titles.append(row_title)

        # use fixed threshold for the visualization panels (what readers expect)
        img = tensor_to_img(x[0])
        draw_triptych(axes[r], img, y[0], (p > args.thresh).float()[0])

        axes[r, 0].set_title(f"Image  (idx={i})", fontsize=9)
        axes[r, 1].set_title("GT (green)", fontsize=9)
        axes[r, 2].set_title(row_title, fontsize=9)

        # record gain if sorting requested
        gain = (best_d - d0) if args.sweep else 0.0
        order.append((gain, r, i))

    if args.sort_by_gain and rows > 1:
        # Reorder rows visually by gain (descending)
        order.sort(key=lambda t: t[0], reverse=True)
        # Re-plot into a new figure with sorted rows
        fig2, axes2 = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if rows == 1: axes2 = np.array([axes2])

        for new_r, (_, old_r, i) in enumerate(order):
            for c in range(cols):
                axes2[new_r, c].imshow(axes[old_r, c].images[0].get_array())
                axes2[new_r, c].axis("off")
            axes2[new_r, 0].set_title(f"Image  (idx={i})", fontsize=9)
            axes2[new_r, 1].set_title("GT (green)", fontsize=9)
            axes2[new_r, 2].set_title(per_row_titles[old_r], fontsize=9)
        plt.tight_layout(h_pad=0.6, w_pad=0.2)
        if args.outfile:
            os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
            plt.savefig(args.outfile, dpi=220)
            print(f"[OK] saved {args.outfile}")
        else:
            plt.show()
    else:
        plt.tight_layout(h_pad=0.6, w_pad=0.2)
        if args.outfile:
            os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
            plt.savefig(args.outfile, dpi=220)
            print(f"[OK] saved {args.outfile}")
        else:
            plt.show()

if __name__ == "__main__":
    main()
