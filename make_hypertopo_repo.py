#!/usr/bin/env python3
import argparse, os, textwrap, stat
from pathlib import Path

FILES = {}

def add(path, content):
    FILES[path] = textwrap.dedent(content).lstrip("\n") + ("\n" if not content.endswith("\n") else "")

# -----------------------------
# README
# -----------------------------
add("README.md", """
# HyperTopo-Adapters: Mixed-Curvature Latents + Topology-Preserving Losses

This repo implements a product-manifold latent head (H⊕E⊕S) atop a frozen tokenizer, with hyperbolic metric learning and topology-preserving regularization (persistent homology) for micro-lesion segmentation.

Highlights:
- Strict **Euclidean-only** baseline (`model.euclidean_only: true`)
- **Qualitative dump** utility (`scripts/dump_quali.py`)
- **Ablation sweep** + **collation to CSV & LaTeX table** (`scripts/collate_results.py`)

See this README and `paper/checklist.md` for a smooth path to a workshop paper.
""")

# -----------------------------
# Requirements
# -----------------------------
add("requirements.txt", """
# Core
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.26
scipy>=1.11
scikit-image>=0.22.0
einops>=0.7.0
tqdm>=4.66.0
pyyaml>=6.0.1
pandas>=2.2.0

# Riemannian / Hyperbolic
geoopt>=0.12.1

# Topology
gudhi>=3.9.0
ripser>=0.6.5

# Viz & utils
matplotlib>=3.8.0
opencv-python>=4.9.0.80
""")

# -----------------------------
# Configs
# -----------------------------
add("configs/default.yaml", """
experiment:
  name: hypertopo_adapters_baseline
  seed: 1337
  precision: bf16

data:
  root: /path/to/lesion_patches
  train_csv: ./splits/train.csv
  val_csv:   ./splits/val.csv
  test_csv:  ./splits/test.csv
  image_key: image_path
  mask_key:  mask_path
  patch_size: [128, 128]
  num_workers: 4

model:
  backbone: tiny_tokenizer
  train_tokenizer: false
  euclidean_only: false           # set true for strict Euclidean baseline
  latent_dims: { dh: 16, de: 32, ds: 16 }
  curvature:  { learn: true, k_h_init: -1.0, k_s_init: 1.0 }

loss:
  dice_weight: 1.0
  bce_weight: 0.5
  hyperbolic_metric_weight: 0.5
  topology_weight: 0.25           # set 0.0 for ablation
  contrastive: { temperature: 0.1 }

optimizer:
  name: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  epochs: 30
  batch_size: 16

logging:
  out_dir: ./outputs
  save_every: 5
  num_vis: 8
""")

add("configs/euclidean_only.yaml", """
# Strict Euclidean baseline: disables hyperbolic & spherical features and losses
experiment: { name: euclidean_only_baseline, seed: 1337, precision: bf16 }
data:
  root: /path/to/lesion_patches
  train_csv: ./splits/train.csv
  val_csv:   ./splits/val.csv
  test_csv:  ./splits/test.csv
  image_key: image_path
  mask_key:  mask_path
  patch_size: [128, 128]
  num_workers: 4
model:
  backbone: tiny_tokenizer
  train_tokenizer: false
  euclidean_only: true
  latent_dims: { dh: 16, de: 32, ds: 16 }   # dh/ds ignored when euclidean_only=true
  curvature:  { learn: false, k_h_init: -1.0, k_s_init: 1.0 }
loss:
  dice_weight: 1.0
  bce_weight: 0.5
  hyperbolic_metric_weight: 0.0   # turn off hyperbolic InfoNCE
  topology_weight: 0.0            # no PH term
  contrastive: { temperature: 0.1 }
optimizer: { name: adamw, lr: 1.0e-3, weight_decay: 1.0e-4, epochs: 30, batch_size: 16 }
logging:   { out_dir: ./outputs_euclidean, save_every: 5, num_vis: 8 }
""")

add("configs/sweep.yaml", """
sweep:
  seeds: [1337, 2025]
  topology_weight: [0.0, 0.25]
  dh: [16, 32]
  ds: [16, 32]
  temperature: [0.07, 0.1, 0.2]
""")

# -----------------------------
# Utils
# -----------------------------
add("src/utils/io.py", """
from dataclasses import dataclass
from typing import Any, Dict
import yaml, os

@dataclass
class Cfg:
    raw: Dict[str, Any]

def load_config(path: str) -> Cfg:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Cfg(raw=cfg)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
""")

add("src/utils/metrics.py", """
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

try:
    import gudhi as gd
    _HAS_GUDHI = True
except Exception:
    _HAS_GUDHI = False

from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.measure import label as cc_label

def _to_numpy(mask: torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return mask

def dice_coeff(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    gt_bin = (gt > 0.5).float()
    inter = (pred_bin * gt_bin).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + gt_bin.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()

def iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    gt_bin = (gt > 0.5).float()
    inter = (pred_bin * gt_bin).sum(dim=(1,2,3))
    union = (pred_bin + gt_bin).clamp(0,1).sum(dim=(1,2,3))
    return ((inter + eps) / (union + eps)).mean()

def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, tolerance: int = 2, eps: float = 1e-7) -> torch.Tensor:
    pred_b = (pred > 0.5).float()
    gt_b = (gt > 0.5).float()
    selem = disk(1)
    def _boundary(x):
        x_np = _to_numpy(x.squeeze(1))
        bmaps = []
        for m in x_np:
            er = binary_erosion(m.astype(bool), selem)
            b = np.logical_xor(m.astype(bool), er)
            bmaps.append(b.astype(np.uint8))
        return torch.from_numpy(np.stack(bmaps)).to(x.device)
    pb, gb = _boundary(pred_b), _boundary(gt_b)
    selem_t = disk(tolerance)
    def _dilate(x):
        x_np = _to_numpy(x)
        outs = []
        for m in x_np:
            outs.append(binary_dilation(m.astype(bool), selem_t).astype(np.uint8))
        return torch.from_numpy(np.stack(outs)).to(x.device).float()
    pb_d, gb_d = _dilate(pb), _dilate(gb)
    tp = (torch.minimum(pb.float(), gb_d)).sum(dim=(1,2))
    fp = (pb.float().sum(dim=(1,2)) - tp)
    fn = (gb.float().sum(dim=(1,2)) - (torch.minimum(gb.float(), pb_d)).sum(dim=(1,2)))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return f1.mean()

def betti_numbers(mask: np.ndarray) -> Tuple[int, int]:
    mask = mask.astype(bool)
    beta0 = cc_label(mask, connectivity=1).max()
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(mask)
    holes = np.logical_and(filled, np.logical_not(mask))
    beta1 = cc_label(holes, connectivity=1).max()
    return int(beta0), int(beta1)

def betti_error(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred_b = (pred > 0.5).float()
    gt_b = (gt > 0.5).float()
    pred_np = _to_numpy(pred_b.squeeze(1))
    gt_np = _to_numpy(gt_b.squeeze(1))
    d0, d1 = [], []
    for p, g in zip(pred_np, gt_np):
        b0_p, b1_p = betti_numbers(p)
        b0_g, b1_g = betti_numbers(g)
        d0.append(abs(b0_p - b0_g))
        d1.append(abs(b1_p - b1_g))
    return {"beta0": float(np.mean(d0)), "beta1": float(np.mean(d1))}

def persistence_diagrams_from_prob(prob: torch.Tensor):
    if not _HAS_GUDHI:
        return [], []
    prob_np = _to_numpy(prob.squeeze(1))
    H0_list, H1_list = [], []
    for img in prob_np:
        f = 1.0 - img.astype(np.float64)
        cc = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=f.flatten())
        _ = cc.persistence(homology_coeff_field=2, persistence_dim_max=True)
        import numpy as np
        d0 = np.array(cc.persistence_intervals_in_dimension(0)).tolist()
        d1 = np.array(cc.persistence_intervals_in_dimension(1)).tolist()
        H0_list.append(d0); H1_list.append(d1)
    return H0_list, H1_list

def pd_distance_batch(pred: torch.Tensor, gt: torch.Tensor, metric: str = "bottleneck") -> float:
    if not _HAS_GUDHI:
        return float("nan")
    try:
        from gudhi.hera import bottleneck_distance
    except Exception:
        bottleneck_distance = None
    try:
        from gudhi.wasserstein import wasserstein_distance
    except Exception:
        wasserstein_distance = None
    H0_p, H1_p = persistence_diagrams_from_prob(pred)
    H0_g, H1_g = persistence_diagrams_from_prob(gt)
    import numpy as np
    dists = []
    for d0p, d0g, d1p, d1g in zip(H0_p, H0_g, H1_p, H1_g):
        if metric == "bottleneck" and bottleneck_distance is not None:
            d0 = bottleneck_distance(d0p, d0g)
            d1 = bottleneck_distance(d1p, d1g)
        elif metric == "wasserstein" and wasserstein_distance is not None:
            d0 = wasserstein_distance(d0p, d0g, order=1., internal_p=2.)
            d1 = wasserstein_distance(d1p, d1g, order=1., internal_p=2.)
        else:
            d0 = np.nan; d1 = np.nan
        dists.append(np.nanmean([d0, d1]))
    return float(np.nanmean(dists))
""")

add("src/utils/training.py", """
from typing import Dict
import torch
from src.utils.metrics import dice_coeff, iou, boundary_f1, betti_error, pd_distance_batch

@torch.no_grad()
def evaluate(model_bundle, val_loader, device='cuda') -> Dict[str, float]:
    tokenizer, head, decoder = model_bundle['tokenizer'], model_bundle['head'], model_bundle['decoder']
    tokenizer.eval(); head.eval(); decoder.eval()
    dices, ious, bndf = [], [], []
    betti0, betti1 = [], []
    pd_dists = []
    for batch in val_loader:
        x, y = batch['image'].to(device), batch['mask'].to(device)
        tokens, hw = tokenizer(x)
        lat = head(tokens)
        pooled = lat['tangent'].mean(dim=1)
        pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
        pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))
        dices.append(dice_coeff(pred, y).item())
        ious.append(iou(pred, y).item())
        bndf.append(boundary_f1(pred, y).item())
        be = betti_error(pred, y)
        betti0.append(be['beta0']); betti1.append(be['beta1'])
        pd_dists.append(pd_distance_batch(pred, y))
    return {
        'dice': float(sum(dices)/len(dices)),
        'iou': float(sum(ious)/len(ious)),
        'boundary_f1': float(sum(bndf)/len(bndf)),
        'betti0_err': float(sum(betti0)/len(betti0)),
        'betti1_err': float(sum(betti1)/len(betti1)),
        'pd_dist': float(sum(pd_dists)/len(pd_dists))
    }
""")

# -----------------------------
# Losses
# -----------------------------
add("src/losses/segmentation.py", """
import torch
import torch.nn.functional as F

def dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.clamp(0,1)
    gt = gt.clamp(0,1)
    inter = (pred * gt).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def bce_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, gt)

def combo_loss(pred: torch.Tensor, gt: torch.Tensor, w_dice=1.0, w_bce=0.5) -> torch.Tensor:
    return w_dice * dice_loss(pred, gt) + w_bce * bce_loss(pred, gt)
""")

add("src/losses/hyperbolic.py", """
import torch
try:
    import geoopt
    HAS_GEOOPT = True
except Exception:
    HAS_GEOOPT = False

def hyperbolic_pairwise_dist(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    if HAS_GEOOPT:
        ball = geoopt.PoincareBall(c=c)
        x_i = x.unsqueeze(1)
        x_j = x.unsqueeze(0)
        d = ball.dist(x_i, x_j)
        return d
    else:
        x2 = (x * x).sum(-1, keepdim=True)
        dist2 = x2 + x2.T - 2 * (x @ x.T)
        dist2 = torch.clamp(dist2, min=0)
        return torch.sqrt(dist2 + 1e-8)

def info_nce_from_pairs(z: torch.Tensor, batch_ids: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    N = z.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=z.device)
    D = hyperbolic_pairwise_dist(z)
    logits = -D / temperature
    mask = torch.eye(N, device=z.device).bool()
    logits = logits.masked_fill(mask, float('-inf'))
    loss = 0.0; count = 0
    for i in range(N):
        pos_idx = (batch_ids == batch_ids[i]).nonzero(as_tuple=False).squeeze(-1)
        pos_idx = pos_idx[pos_idx != i]
        if len(pos_idx) == 0: continue
        logp = torch.log_softmax(logits[i], dim=-1)
        pos_logp = logp[pos_idx].logsumexp(dim=0) - torch.log(torch.tensor(len(pos_idx), device=z.device, dtype=logp.dtype))
        loss = loss - pos_logp; count += 1
    loss = loss / max(count, 1)
    return loss
""")

add("src/losses/topology.py", "# PD distance & Betti metrics live in src/utils/metrics.py\n")

# -----------------------------
# Models
# -----------------------------
add("src/models/backbone.py", """
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTokenizer(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, x):
        f = self.conv(x)
        B, C, H8, W8 = f.shape
        tokens = f.view(B, C, H8*W8).transpose(1,2)
        return tokens, (H8, W8)

class SimpleDecoder(nn.Module):
    def __init__(self, in_dim: int, out_hw: tuple):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_hw[0]*out_hw[1]),
        )
        self.out_hw = out_hw
    def forward(self, pooled, upsample_to: tuple):
        B = pooled.shape[0]
        logits_low = self.fc(pooled).view(B, 1, *self.out_hw)
        logits = F.interpolate(logits_low, size=upsample_to, mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)
""")

add("src/models/manifold.py", """
from typing import Dict, Optional
import torch
import torch.nn as nn

try:
    import geoopt
    HAS_GEOOPT = True
except Exception:
    HAS_GEOOPT = False

class ProductManifoldHead(nn.Module):
    def __init__(self, in_dim: int, dh: int, de: int, ds: int, learn_curvature: bool = True, euclidean_only: bool = False):
        super().__init__()
        self.proj_h = nn.Linear(in_dim, dh)
        self.proj_e = nn.Linear(in_dim, de)
        self.proj_s = nn.Linear(in_dim, ds)
        self.learn_curvature = learn_curvature
        self.euclidean_only = euclidean_only
        self.raw_kh = nn.Parameter(torch.tensor(1.0)) if learn_curvature else None
        self.raw_ks = nn.Parameter(torch.tensor(1.0)) if learn_curvature else None
        if HAS_GEOOPT:
            self.ball = geoopt.PoincareBall(c=1.0)

    def curvatures(self) -> Dict[str, torch.Tensor]:
        if not self.learn_curvature:
            return {"kh": torch.tensor(-1.0), "ks": torch.tensor(1.0)}
        kh = torch.nn.functional.softplus(self.raw_kh) + 1e-6
        ks = torch.nn.functional.softplus(self.raw_ks) + 1e-6
        return {"kh": -kh, "ks": ks}

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        e = self.proj_e(x)
        if self.euclidean_only:
            concat_tangent = e
            return {"latent_H": None, "latent_E": e, "latent_S": None, "tangent": concat_tangent, "k": self.curvatures()}

        h_tan = self.proj_h(x)
        s_tan = self.proj_s(x)
        if HAS_GEOOPT:
            h = self.ball.expmap0(h_tan)
            s = s_tan / (s_tan.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            h = torch.tanh(h_tan)
            s = torch.nn.functional.normalize(s_tan, dim=-1)
        concat_tangent = torch.cat([h_tan, e, s_tan], dim=-1)
        return {"latent_H": h, "latent_E": e, "latent_S": s, "tangent": concat_tangent, "k": self.curvatures()}
""")

# -----------------------------
# Data
# -----------------------------
add("src/data/lesion_dataset.py", """
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class LesionDataset(Dataset):
    def __init__(self, csv_path: str, image_key: str, mask_key: str, size=(128,128)):
        self.df = pd.read_csv(csv_path)
        self.image_key = image_key
        self.mask_key = mask_key
        self.size = size
    def __len__(self):
        return len(self.df)
    def _load_img(self, path: str) -> np.ndarray:
        im = Image.open(path).convert("RGB").resize(self.size, resample=Image.BILINEAR)
        return np.asarray(im)
    def _load_mask(self, path: str) -> np.ndarray:
        mk = Image.open(path).convert("L").resize(self.size, resample=Image.NEAREST)
        mk = (np.asarray(mk) > 127).astype(np.float32)
        return mk
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_img(row[self.image_key'])
        y = self._load_mask(row[self.mask_key'])
        x = torch.from_numpy(x).permute(2,0,1).float() / 255.0
        y = torch.from_numpy(y)[None, ...].float()
        return {"image": x, "mask": y}
""".replace("self.image_key']", "self.image_key]").replace("self.mask_key']", "self.mask_key]"))

# -----------------------------
# Train / Eval
# -----------------------------
add("src/train.py", """
import argparse, os, json
import torch
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder
from src.losses.hyperbolic import info_nce_from_pairs
from src.losses.segmentation import combo_loss
from src.utils.training import evaluate

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config).raw
    device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)

    # Data
    dcfg = cfg['data']
    train_ds = LesionDataset(dcfg['train_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    val_ds   = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    train_loader = DataLoader(train_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=True, num_workers=dcfg['num_workers'])
    val_loader   = DataLoader(val_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])

    # Model
    tokenizer = TinyTokenizer(out_dim=256).to(device)
    eu_only = cfg['model'].get('euclidean_only', False)
    head = ProductManifoldHead(in_dim=256,
                               dh=cfg['model']['latent_dims']['dh'],
                               de=cfg['model']['latent_dims']['de'],
                               ds=cfg['model']['latent_dims']['ds'],
                               learn_curvature=cfg['model']['curvature']['learn'],
                               euclidean_only=eu_only).to(device)
    in_dim = (256 + cfg['model']['latent_dims']['de']) if eu_only else (256 + sum(cfg['model']['latent_dims'].values()))
    decoder = SimpleDecoder(in_dim=in_dim, out_hw=(16,16)).to(device)

    params = list(head.parameters()) + list(decoder.parameters())
    if cfg['model'].get('train_tokenizer', False): params += list(tokenizer.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])

    if args.dry_run:
        print("Dry run successful: data+model+optimizer instantiated."); return

    w_dice, w_bce = cfg['loss']['dice_weight'], cfg['loss'].get('bce_weight', 0.5)
    w_hyp, w_topo = cfg['loss']['hyperbolic_metric_weight'], cfg['loss']['topology_weight']
    temp = cfg['loss']['contrastive']['temperature']

    best = {"dice": 0.0}
    for epoch in range(cfg['optimizer']['epochs']):
        tokenizer.train(False); head.train(); decoder.train()
        for batch in train_loader:
            x = batch['image'].to(device); y = batch['mask'].to(device)
            tokens, hw = tokenizer(x)
            lat = head(tokens)
            pooled = lat['tangent'].mean(dim=1)
            pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
            pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))

            # Segmentation loss
            loss_seg = combo_loss(pred, y, w_dice, w_bce)

            # Hyperbolic metric loss (skip if Euclidean-only)
            loss_hyp = torch.tensor(0.0, device=device)
            if (not eu_only) and (w_hyp > 0.0) and (lat['latent_H'] is not None):
                B, T, _ = lat['latent_H'].shape
                z = lat['latent_H'].reshape(B*T, -1)
                batch_ids = torch.arange(B, device=z.device).repeat_interleave(T)
                loss_hyp = info_nce_from_pairs(z, batch_ids, temperature=temp) * w_hyp

            # Topology term (optional; can be heavy)
            from src.utils.metrics import pd_distance_batch
            pd_dist = pd_distance_batch(pred.detach(), y.detach())
            loss_topo = (torch.tensor(pd_dist, device=device) if pd_dist == pd_dist else torch.tensor(0.0, device=device)) * w_topo

            loss = loss_seg + loss_hyp + loss_topo
            opt.zero_grad(); loss.backward(); opt.step()

        metrics = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, val_loader, device=device)
        print(f"Epoch {epoch+1}/{cfg['optimizer']['epochs']} | val: {metrics}")
        if metrics['dice'] > best['dice']:
            best = metrics.copy()
            with open(os.path.join(out_dir, 'best_metrics.json'), 'w') as f: json.dump(best, f, indent=2)
    print("Training complete. Best val:", best)

if __name__ == '__main__':
    main()
""")

add("src/eval.py", """
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder
from src.utils.training import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = load_config(args.config).raw; device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)
    dcfg = cfg['data']
    val_ds = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    test_ds= LesionDataset(dcfg['test_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    val_loader = DataLoader(val_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])
    test_loader= DataLoader(test_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])
    tokenizer = TinyTokenizer(out_dim=256).to(device)
    eu_only = cfg['model'].get('euclidean_only', False)
    head = ProductManifoldHead(in_dim=256,
                               dh=cfg['model']['latent_dims']['dh'],
                               de=cfg['model']['latent_dims']['de'],
                               ds=cfg['model']['latent_dims']['ds'],
                               learn_curvature=cfg['model']['curvature']['learn'],
                               euclidean_only=eu_only).to(device)
    in_dim = (256 + cfg['model']['latent_dims']['de']) if eu_only else (256 + sum(cfg['model']['latent_dims'].values()))
    decoder = SimpleDecoder(in_dim=in_dim, out_hw=(16,16)).to(device)
    metrics_val = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, val_loader, device=device)
    metrics_test = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, test_loader, device=device)
    with open(os.path.join(out_dir, 'metrics_val.json'), 'w') as f: json.dump(metrics_val, f, indent=2)
    with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f: json.dump(metrics_test, f, indent=2)
    print('VAL:', metrics_val); print('TEST:', metrics_test)

if __name__ == '__main__':
    main()
""")

# -----------------------------
# Scripts
# -----------------------------
add("scripts/run_train.sh", """
#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/default.yaml}
python -m src.train --config ${CFG} --dry_run
python -m src.train --config ${CFG}
python -m src.eval  --config ${CFG}
""")

add("scripts/make_splits.py", """
import argparse, csv, os, glob, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root with images/ and masks/')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--suffix_img', default='.png')
    ap.add_argument('--suffix_mask', default='.png')
    ap.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    args = ap.parse_args()
    root = Path(args.root)
    imgs = sorted(glob.glob(str(root / 'images' / f'*{args.suffix_img}')))
    rows = []
    for img in imgs:
        stem = Path(img).stem
        mask = str(root / 'masks' / f'{stem}{args.suffix_mask}')
        if os.path.exists(mask):
            rows.append({'image_path': img, 'mask_path': mask})
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n*args.split[0]); n_val = int(n*args.split[1])
    splits = [('train.csv', rows[:n_train]), ('val.csv', rows[n_train:n_train+n_val]), ('test.csv', rows[n_train+n_val:])]
    os.makedirs(args.outdir, exist_ok=True)
    for name, data in splits:
        with open(Path(args.outdir)/name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image_path','mask_path'])
            w.writeheader(); w.writerows(data)
        print('Wrote', name, len(data), 'rows')

if __name__ == '__main__':
    main()
""")

add("scripts/run_sweep.py", """
import yaml, os, copy, subprocess, sys

def run(cfg_path, overrides, log_dir):
    with open(cfg_path, 'r') as f:
        base = yaml.safe_load(f)
    cfg = copy.deepcopy(base)
    for k,v in overrides.items():
        keys = k.split('.'); d = cfg
        for kk in keys[:-1]: d = d[kk]
        d[keys[-1]] = v
    cfg['logging']['out_dir'] = os.path.join(log_dir, 'run_' + '_'.join(f"{k.replace('.','-')}-{v}" for k,v in overrides.items()))
    os.makedirs(cfg['logging']['out_dir'], exist_ok=True)
    tmp_cfg = os.path.join(cfg['logging']['out_dir'], 'config.yaml')
    with open(tmp_cfg, 'w') as f: yaml.safe_dump(cfg, f)
    print('Launching', tmp_cfg)
    subprocess.run([sys.executable, '-m', 'src.train', '--config', tmp_cfg], check=False)

def main():
    with open('configs/default.yaml', 'r') as f: base = yaml.safe_load(f)
    with open('configs/sweep.yaml', 'r') as f: sweep = yaml.safe_load(f)
    log_dir = os.path.join(base['logging']['out_dir'], 'sweeps'); os.makedirs(log_dir, exist_ok=True)
    grid = []
    for tw in sweep['sweep']['topology_weight']:
        for dh in sweep['sweep']['dh']:
            for ds in sweep['sweep']['ds']:
                for temp in sweep['sweep']['temperature']:
                    grid.append({'loss.topology_weight': tw, 'model.latent_dims.dh': dh, 'model.latent_dims.ds': ds, 'loss.contrastive.temperature': temp})
    for run_overrides in grid: run('configs/default.yaml', run_overrides, log_dir)

if __name__ == '__main__':
    main()
""")

add("scripts/dump_quali.py", """
import argparse, os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder

def save_triplet(img, gt, pred, path):
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(img.transpose(1,2,0)); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(gt[0], cmap='gray');   axs[1].set_title('GT'); axs[1].axis('off')
    axs[2].imshow(pred[0], cmap='gray'); axs[2].set_title('Pred'); axs[2].axis('off')
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num', type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    device = args.device
    out_dir = os.path.join(cfg['logging']['out_dir'], 'qualitative'); ensure_dir(out_dir)

    dcfg = cfg['data']
    val_ds = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=dcfg['num_workers'])

    tokenizer = TinyTokenizer(out_dim=256).to(device)
    eu_only = cfg['model'].get('euclidean_only', False)
    head = ProductManifoldHead(in_dim=256,
                               dh=cfg['model']['latent_dims']['dh'],
                               de=cfg['model']['latent_dims']['de'],
                               ds=cfg['model']['latent_dims']['ds'],
                               learn_curvature=cfg['model']['curvature']['learn'],
                               euclidean_only=eu_only).to(device)
    in_dim = (256 + cfg['model']['latent_dims']['de']) if eu_only else (256 + sum(cfg['model']['latent_dims'].values()))
    decoder = SimpleDecoder(in_dim=in_dim, out_hw=(16,16)).to(device)

    tokenizer.eval(); head.eval(); decoder.eval()
    count = 0
    for batch in loader:
        x, y = batch['image'].to(device), batch['mask'].to(device)
        tokens, hw = tokenizer(x)
        lat = head(tokens)
        pooled = lat['tangent'].mean(dim=1)
        pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
        pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))
        img_np  = x[0].detach().cpu().numpy()
        gt_np   = y[0].detach().cpu().numpy()
        pred_np = (pred[0].detach().cpu().numpy() > 0.5).astype(np.float32)
        save_triplet(img_np, gt_np, pred_np, os.path.join(out_dir, f'sample_{count:03d}.png'))
        count += 1
        if count >= args.num: break
    print('Saved', count, 'samples to', out_dir)

if __name__ == '__main__':
    main()
""")

# Collation: turn many run_* best_metrics.json into CSV + LaTeX
add("scripts/collate_results.py", """
import argparse, os, glob, json, yaml, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/sweeps')
    ap.add_argument('--out_csv', default='outputs/sweeps/summary.csv')
    ap.add_argument('--latex', default='outputs/sweeps/table.tex')
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, 'run_*'))):
        cfgp = os.path.join(d, 'config.yaml')
        metp = os.path.join(d, 'best_metrics.json')
        if not (os.path.exists(cfgp) and os.path.exists(metp)):
            continue
        with open(cfgp, 'r') as f: cfg = yaml.safe_load(f)
        with open(metp, 'r') as f: met = json.load(f)
        row = {
            'dir': d,
            'topology_weight': cfg['loss']['topology_weight'],
            'dh': cfg['model']['latent_dims']['dh'],
            'ds': cfg['model']['latent_dims']['ds'],
            'temp': cfg['loss']['contrastive']['temperature'],
            'dice': met.get('dice'),
            'iou': met.get('iou'),
            'boundary_f1': met.get('boundary_f1'),
            'betti0_err': met.get('betti0_err'),
            'betti1_err': met.get('betti1_err'),
            'pd_dist': met.get('pd_dist'),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows: w.writeheader(); w.writerows(rows)
    print('Wrote CSV:', args.out_csv)

    # Minimal LaTeX table
    with open(args.latex, 'w') as f:
        f.write('\\begin{tabular}{lcccccc}\\n')
        f.write('\\toprule\\n')
        f.write('Setting & Dice $\\uparrow$ & IoU $\\uparrow$ & B-F1 $\\uparrow$ & $\\Delta\\beta_0$ $\\downarrow$ & $\\Delta\\beta_1$ $\\downarrow$ & PD $\\downarrow$ \\\\\\n')
        f.write('\\midrule\\n')
        for r in rows:
            setting = f"tw={r['topology_weight']}, dh={r['dh']}, ds={r['ds']}, T={r['temp']}"
            f.write(f"{setting} & {r['dice']:.3f} & {r['iou']:.3f} & {r['boundary_f1']:.3f} & {r['betti0_err']:.2f} & {r['betti1_err']:.2f} & {r['pd_dist'] if r['pd_dist'] is not None else 'NaN'} \\\\\\n")
        f.write('\\bottomrule\\n')
        f.write('\\end{tabular}\\n')
    print('Wrote LaTeX table:', args.latex)

if __name__ == '__main__':
    main()
""")

# -----------------------------
# Paper & tests
# -----------------------------
add("paper/main.tex", r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\title{HyperTopo-Adapters: Product-Manifold Latents with Topological Priors for Micro-Lesion Segmentation}
\author{Anonymous Authors}
\begin{document}
\maketitle
\begin{abstract}
Foundation-model vision features are hierarchical, yet Euclidean decoders often distort fine-scale topology in biomedical segmentation. We propose a product-manifold latent head (hyperbolic $\oplus$ Euclidean $\oplus$ spherical) with hyperbolic metric learning and topology-preserving regularization via persistent homology.
\end{abstract}
\section{Introduction}\section{Related Work}\section{Method}\section{Experiments}\section{Discussion}
\bibliographystyle{unsrt}\bibliography{refs}
\end{document}
""")

add("paper/outline.md", "# Keep this synced with your latest outline & figures.\n")
add("paper/checklist.md", "- [ ] Double-blind\n- [ ] ≤ 9 pages main\n- [ ] Reproducibility\n- [ ] Ethics/Broader Impact\n- [ ] Vector figures\n")

add("tests/test_imports.py", """
def test_dummy():
    import src.models.manifold as M
    import src.losses.hyperbolic as H
    import src.losses.topology as T
    import src.data.lesion_dataset as D
    assert hasattr(M, 'ProductManifoldHead')
    assert hasattr(H, 'info_nce_from_pairs')
    assert hasattr(D, 'LesionDataset')
""")

# -----------------------------
# Writer
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="hypertopo_adapters_repo_full")
    args = ap.parse_args()
    dest = Path(args.dest)
    for p, content in FILES.items():
        out = dest / p
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
    # make script executable
    run_sh = dest / "scripts/run_train.sh"
    if run_sh.exists():
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IEXEC)
    print(f"Wrote {len(FILES)} files into {dest}")
    print("Next:")
    print(f"  cd {dest}")
    print("  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")

if __name__ == "__main__":
    main()
