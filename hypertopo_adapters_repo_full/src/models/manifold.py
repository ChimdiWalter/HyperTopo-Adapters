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
