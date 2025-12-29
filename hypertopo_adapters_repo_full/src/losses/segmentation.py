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
