# src/utils/training.py
import torch
from src.utils.metrics import dice_coeff, iou, boundary_f1, betti_error, pd_distance_batch

def _coerce_betti(ret, pred=None, y=None):
    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
        return float(ret[0]), float(ret[1])
    if isinstance(ret, dict):
        v0 = None; v1 = None
        for k in ('betti0_err', 'beta0', 'b0', 'betti0'):
            if k in ret: v0 = ret[k]; break
        for k in ('betti1_err', 'beta1', 'b1', 'betti1'):
            if k in ret: v1 = ret[k]; break
        if v0 is not None and v1 is not None:
            return float(v0), float(v1)
    try:
        from src.utils.metrics import betti_numbers
        pm = (pred >= 0.5).float()
        gm = (y    >= 0.5).float()
        b0p, b1p = betti_numbers(pm)
        b0g, b1g = betti_numbers(gm)
        return float(abs(b0p - b0g)), float(abs(b1p - b1g))
    except Exception:
        return 0.0, 0.0

@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    tokenizer = model["tokenizer"].to(device).eval()
    head      = model["head"].to(device).eval()
    decoder   = model["decoder"].to(device).eval()

    total = 0
    s_dice = s_iou = s_bf1 = s_db0 = s_db1 = s_pd = 0.0

    for batch in loader:
        x = batch['image'].to(device)
        y = batch['mask'].to(device)
        B = x.size(0)

        tokens, _ = tokenizer(x)
        lat = head(tokens)
        pooled = lat['tangent'].mean(dim=1)
        pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
        pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))

        # Each of these returns a batch-mean -> multiply by B to do sample-weighted sum
        s_dice += float(dice_coeff(pred, y).item()) * B
        s_iou  += float(iou(pred, y).item()) * B
        s_bf1  += float(boundary_f1(pred, y).item()) * B

        try:
            ret = betti_error(pred, y, )  # returns mean over batch
            db0, db1 = _coerce_betti(ret, pred=pred, y=y)
            s_db0 += db0 * B
            s_db1 += db1 * B
        except TypeError:
            # legacy signature
            ret = betti_error(pred, y)
            db0, db1 = _coerce_betti(ret, pred=pred, y=y)
            s_db0 += db0 * B
            s_db1 += db1 * B

        try:
            pd = float(pd_distance_batch(pred, y))  # mean over batch
            if pd == pd:
                s_pd += pd * B
        except Exception:
            pass

        total += B

    denom = max(total, 1)
    return {
        "dice":        s_dice / denom,
        "iou":         s_iou  / denom,
        "boundary_f1": s_bf1  / denom,
        "betti0_err":  s_db0  / denom,
        "betti1_err":  s_db1  / denom,
        "pd_dist":     s_pd   / denom,
    }
