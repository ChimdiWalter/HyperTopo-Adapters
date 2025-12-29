# src/models/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder, flatten_tokens

# -------------------- Your original tiny CNN tokenizer --------------------
class TinyTokenizer(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, x):
        f = self.conv(x)  # [B,C,H/8,W/8]
        B, C, H8, W8 = f.shape
        tokens = f.view(B, C, H8*W8).transpose(1,2)  # [B,T,C]
        return tokens, (H8, W8)

# -------------------- Simple pooled decoder (keep for ablation) --------------------
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



# ---- NEW: wrapper with selective unfreezing ----
class FrozenBackboneTokenizer(nn.Module):
    def __init__(self, backbone_name: str, de: int, freeze: bool = True, unfreeze_keys=None):
        super().__init__()
        self.encoder = build_encoder(backbone_name, proj_dim=de, pretrained=True)
        self.unfreeze_keys = list(unfreeze_keys or [])

        # freeze all by default
        for p in self.encoder.parameters():
            p.requires_grad = False

        # selectively unfreeze: any param whose name contains one of unfreeze_keys
        if freeze:
            for n, p in self.encoder.named_parameters():
                if any(k in n for k in self.unfreeze_keys):
                    p.requires_grad = True
        else:
            # train_tokenizer=True means train everything
            for p in self.encoder.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = self.encoder(x)   # {'feat': [B,de,H',W']}
        feat = out["feat"]
        B, C, Hp, Wp = feat.shape
        tokens = flatten_tokens(feat)  # [B, T, de]
        return tokens, (Hp, Wp)

def make_tokenizer(backbone_name: str, de: int, train_tokenizer, unfreeze_keys=None):
    # train_tokenizer can be: False | True | 'partial'
    if backbone_name is None or backbone_name.lower() == "tiny_tokenizer":
        return TinyTokenizer(out_dim=de)

    if train_tokenizer is True:
        return FrozenBackboneTokenizer(backbone_name, de=de, freeze=False, unfreeze_keys=unfreeze_keys)

    if isinstance(train_tokenizer, str) and train_tokenizer.lower() == "partial":
        # freeze most, unfreeze selected keys
        return FrozenBackboneTokenizer(backbone_name, de=de, freeze=True, unfreeze_keys=unfreeze_keys)

    # default: fully frozen
    return FrozenBackboneTokenizer(backbone_name, de=de, freeze=True, unfreeze_keys=None)



# -------------------- NEW: spatial decoder (per-token → low-res map → upsample) --------------------
class UpBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)



class SpatialDecoder(nn.Module):
    """
    Token-grid-aware decoder:
      - Saves low-res grid (Hp, Wp) at init
      - Accepts optional low_hw in forward (defaults to saved one)
      - Projects pooled vector to a [C0, Hp, Wp] map, then upsamples
    """
    def __init__(self, in_dim: int, low_hw: tuple, max_upsamples: int = None):
        super().__init__()
        Hp, Wp = low_hw
        self.low_hw = (Hp, Wp)
        self.C0 = 64

        # *** KEY FIX: project to C0 * Hp * Wp, not just C0 ***
        self.proj = nn.Linear(in_dim, self.C0 * Hp * Wp)

        def up_block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.head = nn.Sequential(
            nn.Conv2d(self.C0, self.C0, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.C0, self.C0, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up2 = up_block(self.C0, self.C0)  # x2
        self.up4 = up_block(self.C0, 32)       # x4
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, pooled, upsample_to: tuple, low_hw: tuple = None):
        if low_hw is None:
            Hp, Wp = self.low_hw
        else:
            Hp, Wp = low_hw

        B = pooled.shape[0]
        f = self.proj(pooled).view(B, self.C0, Hp, Wp)
        f = self.head(f)
        f = self.up2(f)  # x2
        f = self.up4(f)  # x4
        f = F.interpolate(f, size=upsample_to, mode='bilinear', align_corners=False)
        logits = self.out(f)
        return torch.sigmoid(logits)
