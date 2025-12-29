#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative exports: image / ground-truth / prediction triptychs.

Usage:
  python qualitative.py \
    --config configs/default.yaml \
    --ckpt outputs/hes_topo/ckpt_minPD_in_topK.pt \
    --split val \
    --num 12 \
    --outfile outputs/qual_val_grid.png

If --ckpt is omitted, the script tries ckpt_minPD_in_topK.pt then best_adapter.pt from cfg.logging.out_dir.
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
from src.utils.training import evaluate  # for consistent metric fns
from src.utils.metrics import dice_coeff, iou, boundary_f1

def choose_ckpt(out_dir: str, arg_ckpt: str | None) -> str | None:
    if arg_ckpt and os.path.isfile(arg_ckpt):
        return arg_ckpt
    p = os.path.join(out_dir, "ckpt_minPD_in_topK.pt")
    if os.path.isfile(p):
        return p
    p = os.path.join(out_dir, "best_adapter.pt")
    if os.path.isfile(p):
        return p
    return None

@torch.no_grad()
def run_model(model, x: torch.Tensor) -> torch.Tensor:
    tokenizer, head, decoder = model["tokenizer"], model["head"], model["decoder"]
    tokens, _ = tokenizer(x)
    lat = head(tokens)
    pooled = lat['tangent'].mean(dim=1)
    pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
    pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))
    return pred

def tensor_to_img(x: torch.Tensor) -> np.ndarray:
    # assume x in [0,1], shape [3,H,W] or [1,H,W]
    x = x.detach().cpu().clamp(0,1).numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    x = (x.transpose(1,2,0) * 255.0).round().astype(np.uint8)
    return x

def mask_to_rgb(m: torch.Tensor, color=(255,0,0), alpha=0.4) -> np.ndarray:
    # m is [1,H,W] in [0,1]
    m = (m.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    H, W = m.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[m == 1] = np.array(color, dtype=np.uint8)
    return overlay, alpha

def draw_triptych(ax, img, gt, pred):
    # left: image, mid: GT overlay, right: pred overlay
    from skimage import measure
    H, W, _ = img.shape
    gt_bin = (gt.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    pr_bin = (pred.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8)

    panels = []
    # image only
    panels.append(img)

    # GT overlay
    gt_rgb, a = mask_to_rgb(gt, color=(180, 105, 255), alpha=0.35)
    gt_vis = (img * (1 - a) + gt_rgb * a).astype(np.uint8)
    panels.append(gt_vis)

    # PR overlay
    pr_rgb, a = mask_to_rgb(pred, color=(255, 0, 0), alpha=0.35)
    pr_vis = (img * (1 - a) + pr_rgb * a).astype(np.uint8)
    panels.append(pr_vis)

    for j, p in enumerate(panels):
        ax[j].imshow(p)
        ax[j].axis("off")

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
    if args.use_simple_decoder:
        decoder = SimpleDecoder(in_dim=dec_in_dim, out_hw=out_hw).to(device)
    else:
        decoder = SpatialDecoder(in_dim=dec_in_dim, low_hw=out_hw).to(device)

    # checkpoint
    out_dir = cfg['logging']['out_dir']
    ckpt_path = choose_ckpt(out_dir, args.ckpt)
    if ckpt_path:
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "head" in ckpt:     head.load_state_dict(ckpt["head"], strict=False)
        if "decoder" in ckpt:  decoder.load_state_dict(ckpt["decoder"], strict=False)
        if "tokenizer" in ckpt: tokenizer.load_state_dict(ckpt["tokenizer"], strict=False)
    else:
        print("[WARN] No checkpoint found; using random head/decoder weights.")

    model = {"tokenizer": tokenizer.eval(), "head": head.eval(), "decoder": decoder.eval()}

    # build grid
    n = len(idxs)
    cols = 3            # triptych per sample
    rows = n
    fig_h = max(4, int(1.6 * rows))
    fig_w = 10
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.array([axes])  # normalize shape

    # render
    for r, i in enumerate(idxs):
        sample = ds[i]
        x = sample["image"].unsqueeze(0).to(device)  # [1,3,H,W]
        y = sample["mask"].unsqueeze(0).to(device)   # [1,1,H,W]
        pred = run_model(model, x)

        # per-image metrics
        d = float(dice_coeff(pred, y).item())
        j = float(iou(pred, y).item())
        bf= float(boundary_f1(pred, y).item())
        title = f"Dice {d:.3f} | IoU {j:.3f} | BF1 {bf:.3f}"

        img = tensor_to_img(x[0])
        draw_triptych(axes[r], img, y[0], pred[0])
        axes[r, 0].set_title(f"Image  (idx={i})", fontsize=9)
        axes[r, 1].set_title("GT (green)", fontsize=9)
        axes[r, 2].set_title(f"Pred (red) — {title}", fontsize=9)

    plt.tight_layout(h_pad=0.6, w_pad=0.2)
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        plt.savefig(args.outfile, dpi=220)
        print(f"[OK] saved {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

