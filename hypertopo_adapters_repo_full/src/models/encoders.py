import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

try:
    import timm
except Exception:
    timm = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):  # x in [0,1]
        return (x - self.mean) / self.std


def flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
    """[B,C,H',W'] -> [B,T,C] with T=H'*W'."""
    return feat.flatten(2).transpose(1, 2).contiguous()


# ------------------------- ResNet-50 -------------------------
class ResNet50Encoder(nn.Module):
    """Return {'feat': [B, proj_dim, H', W']} from layer4 (stride ~32)."""
    def __init__(self, pretrained: bool = True, proj_dim: int | None = None):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        self.body = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        self.norm = Normalize()
        in_ch = 2048
        self.proj = nn.Identity() if (proj_dim is None or proj_dim == in_ch) else nn.Conv2d(in_ch, proj_dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        f = self.body(x)["feat"]      # [B,2048,H',W']
        f = self.proj(f)              # [B,proj_dim,H',W']
        return {"feat": f}


# --------------------- DeepLabV3-ResNet50 --------------------
class DeepLabV3R50Encoder(nn.Module):
    """Return ASPP backbone out: {'feat': [B, proj_dim, H', W']} (stride ~16)."""
    def __init__(self, pretrained: bool = True, proj_dim: int | None = None):
        super().__init__()
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        dl = models.segmentation.deeplabv3_resnet50(weights=weights)
        self.backbone = dl.backbone                     # returns {'out': [B,2048,H',W']}
        self.norm = Normalize()
        in_ch = 2048
        self.proj = nn.Identity() if (proj_dim is None or proj_dim == in_ch) else nn.Conv2d(in_ch, proj_dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        f = self.backbone(x)["out"]   # [B,2048,H',W']
        f = self.proj(f)              # [B,proj_dim,H',W']
        return {"feat": f}


# -------------------- ViT DINO(v2/v3) via timm --------------------
class ViT_DINO_Encoder(nn.Module):
    """
    DINO ViT with dynamic input: auto-pads HxW to multiples of patch size.
    model_name examples:
      - 'vit_small_patch14_dinov2'  or 'vit_small_patch14_dinov2.lvd142m'
      - 'vit_base_patch14_dinov2'   or 'vit_base_patch14_dinov2.lvd142m'
      - 'vit_base_patch14_dinov3'   (if available in your timm)
    Returns {'feat': [B, proj_dim, H', W']} where H' = ceil(H/patch), W' = ceil(W/patch).
    """
    def __init__(self, model_name: str, pretrained: bool = True, proj_dim: int | None = None):
        super().__init__()
        assert timm is not None, "Please install timm: pip install timm"
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")

        # patch size
        if hasattr(self.net, "patch_embed") and hasattr(self.net.patch_embed, "patch_size"):
            p = self.net.patch_embed.patch_size
            self.patch = p[0] if isinstance(p, (tuple, list)) else int(p)
        else:
            self.patch = int(self.net.default_cfg.get("patch_size", 14))

        # normalization
        mean = self.net.default_cfg.get("mean", IMAGENET_MEAN)
        std  = self.net.default_cfg.get("std",  IMAGENET_STD)
        self.norm = Normalize(mean, std)

        # out channels & projection
        embed_dim = getattr(self.net, "embed_dim", None) or getattr(self.net, "num_features", None)
        out_ch = embed_dim if (proj_dim is None or proj_dim == embed_dim) else proj_dim
        self.proj = nn.Identity() if out_ch == embed_dim else nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False)
        self.out_ch = out_ch

    def _set_dynamic_patch_embed(self, H2: int, W2: int):
        """Update timm PatchEmbed to padded H2xW2 so grid & num_patches are consistent."""
        pe = getattr(self.net, "patch_embed", None)
        if pe is None:
            return
        # patch size (ph,pw)
        ps = pe.patch_size if isinstance(pe.patch_size, (tuple, list)) else (int(pe.patch_size), int(pe.patch_size))
        ph, pw = int(ps[0]), int(ps[1])
        gh, gw = H2 // ph, W2 // pw
        if hasattr(pe, "img_size"):
            pe.img_size = (H2, W2)
        if hasattr(pe, "grid_size"):
            pe.grid_size = (gh, gw)
        if hasattr(pe, "num_patches"):
            pe.num_patches = gh * gw

    def _tokens_to_grid(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """[B,N,C] -> [B,C,gh,gw] (drops prefix/cls tokens if present)."""
        B, N, C = tokens.shape
        num_prefix = getattr(self.net, "num_prefix_tokens", 1)
        if N == gh * gw + num_prefix:
            tokens = tokens[:, num_prefix:, :]
            N = gh * gw
        elif N != gh * gw:
            # last fallback: infer square if possible
            g = int(math.sqrt(N))
            if g * g == N:
                gh, gw = g, g
            else:
                raise RuntimeError(f"Cannot reshape tokens of length {N} to grid {gh}x{gw}")
        grid = tokens.view(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()
        return grid

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        x = self.norm(x)

        # --- pad to multiples of patch size (e.g., 14) ---
        ph = pw = self.patch
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right, bottom
        H2, W2 = H + pad_h, W + pad_w

        # update ViT's internal grid for new H2xW2
        self._set_dynamic_patch_embed(H2, W2)

        # forward to patch tokens
        feats = self.net.forward_features(x)
        if isinstance(feats, dict):
            for k in ("x", "last_hidden_state", "tokens"):
                if k in feats:
                    feats = feats[k]
                    break

        # reshape tokens to grid
        gh, gw = H2 // ph, W2 // pw
        if feats.dim() == 3:
            grid = self._tokens_to_grid(feats, gh, gw)  # [B, embed_dim, gh, gw]
        elif feats.dim() == 4:
            grid = feats  # already [B,C,gh,gw]
        else:
            raise RuntimeError(f"Unexpected ViT features shape: {feats.shape}")

        # project channels if needed
        grid = self.proj(grid)  # [B, out_ch, gh, gw]
        return {"feat": grid}


# ----------------------------- factory -----------------------------
def build_encoder(name: str, proj_dim: int, pretrained: bool = True) -> nn.Module:
    n = (name or "").lower()

    # CNNs
    if n == "resnet50":
        return ResNet50Encoder(pretrained=pretrained, proj_dim=proj_dim)
    if n in ("deeplabv3_r50", "deeplab3_r50"):
        return DeepLabV3R50Encoder(pretrained=pretrained, proj_dim=proj_dim)

    # ViTs (DINO)
    # Use the names available in your timm; both bare and ".lvd142m" variants are common.
    if n in ("vit_dinov2_s14", "dinov2_s14"):
        model_name = "vit_small_patch14_dinov2"
        try:
            # prefer the larger pretrain tag if present
            timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
            model_name = "vit_small_patch14_dinov2.lvd142m"
        except Exception:
            pass
        return ViT_DINO_Encoder(model_name, pretrained=pretrained, proj_dim=proj_dim)

    if n in ("vit_dinov2_b14", "dinov2_b14"):
        model_name = "vit_base_patch14_dinov2"
        try:
            timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
            model_name = "vit_base_patch14_dinov2.lvd142m"
        except Exception:
            pass
        return ViT_DINO_Encoder(model_name, pretrained=pretrained, proj_dim=proj_dim)

    if n in ("vit_dinov3_b14", "dinov3_b14"):
        return ViT_DINO_Encoder("vit_base_patch14_dinov3", pretrained=pretrained, proj_dim=proj_dim)

    raise ValueError(f"Unknown backbone '{name}'")
