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
    # Returns ([H0_diagrams_per_image], [H1_diagrams_per_image])
    if not _HAS_GUDHI:
        return [], []
    prob_np = _to_numpy(prob.squeeze(1))
    H0_list, H1_list = [], []
    for img in prob_np:
        f = 1.0 - img.astype(np.float64)  # low values = foreground
        cc = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=f.flatten())
        # Works across GUDHI versions:
        try:
            cc.persistence(homology_coeff_field=2)
        except TypeError:
            cc.persistence()  # very old versions
        d0 = np.asarray(cc.persistence_intervals_in_dimension(0)).tolist()
        d1 = np.asarray(cc.persistence_intervals_in_dimension(1)).tolist()
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
