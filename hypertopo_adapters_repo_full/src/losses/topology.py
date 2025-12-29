# src/losses/topology.py
import torch
import torch.nn.functional as F

__all__ = [
    "soft_euler_characteristic",
    "hard_euler_characteristic",
    "euler_characteristic_loss",
    "total_variation_loss",
    "topology_surrogate_loss",
]

# ---------- helpers ----------
def _soft_step(p: torch.Tensor, tau: float, sigma: float) -> torch.Tensor:
    """Smooth threshold around tau using a logistic (sigmoid) with width sigma."""
    return torch.sigmoid((p - tau) / max(sigma, 1e-6))

def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure [B,1,H,W]."""
    if x.dim() == 4:
        return x if x.size(1) == 1 else x.mean(dim=1, keepdim=True)
    if x.dim() == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")

# ---------- Euler characteristic (soft) ----------
def soft_euler_characteristic(occ: torch.Tensor) -> torch.Tensor:
    """
    Expected Euler characteristic χ ≈ Faces - Edges + Vertices on a 2D grid (4-neighborhood).
    occ in [0,1], shape [B,1,H,W]. Returns [B] (one χ per image).
    """
    x = _to_bchw(occ).clamp(0, 1)
    # Faces
    Ff = x.sum(dim=(1, 2, 3))
    # Edges
    Eh = (x[:, :, :, :-1] * x[:, :, :, 1:]).sum(dim=(1, 2, 3))
    Ev = (x[:, :, :-1, :] * x[:, :, 1:, :]).sum(dim=(1, 2, 3))
    Ee = Eh + Ev
    # Vertices (2x2 cliques)
    Vv = (
        x[:, :, :-1, :-1]
        * x[:, :, :-1, 1:]
        * x[:, :, 1:, :-1]
        * x[:, :, 1:, 1:]
    ).sum(dim=(1, 2, 3))
    return Ff - Ee + Vv  # [B]

def hard_euler_characteristic(mask: torch.Tensor) -> torch.Tensor:
    """Euler characteristic via the same formula for a binary mask."""
    m = _to_bchw((mask > 0.5).float())
    return soft_euler_characteristic(m)

def euler_characteristic_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thresholds=(0.5,),
    sigma: float = 0.1,
    normalize: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L_EC = avg_{tau} ( χ_soft(sigmoid(p; tau, sigma)) - χ(gt) )^2
    """
    p = _to_bchw(pred).clamp(0, 1)
    g = _to_bchw(gt).clamp(0, 1)
    with torch.no_grad():
        chi_gt = hard_euler_characteristic(g)  # [B]

    B, _, H, W = p.shape
    area = float(H * W)
    losses = []
    for tau in thresholds:
        occ = _soft_step(p, tau=float(tau), sigma=float(sigma))
        chi_pred = soft_euler_characteristic(occ)  # [B]
        se = (chi_pred - chi_gt) ** 2
        if normalize and area > 0:
            se = se / area
        losses.append(se)

    loss = torch.stack(losses, dim=0).mean(dim=0)  # [B]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

# ---------- TV regularizer ----------
def total_variation_loss(pred: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Isotropic TV on probabilities in [0,1]; encourages clean, closed boundaries.
    """
    p = _to_bchw(pred).clamp(0, 1)
    dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
    dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
    tv = dx.mean() + dy.mean() if reduction == "mean" else dx.sum() + dy.sum()
    return tv

# ---------- Combined surrogate (EC + TV) ----------
def topology_surrogate_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thresholds=(0.3, 0.5, 0.7),
    sigma: float = 0.1,
    alpha: float = 1e-3,
    beta:  float = 1e-3,
    normalize: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L_topo = L_EC(pred, gt) + beta * TV(pred), with small alpha/beta scaling available.
      - alpha: (optional) extra multiplier for L_EC if you want it smaller/larger
      - beta:  TV weight
    """
    lec = euler_characteristic_loss(
        pred, gt, thresholds=thresholds, sigma=sigma,
        normalize=normalize, reduction=reduction
    )
    tv  = total_variation_loss(pred, reduction=reduction)
    return alpha * lec + beta * tv
