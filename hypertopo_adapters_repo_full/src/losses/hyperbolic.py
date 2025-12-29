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
