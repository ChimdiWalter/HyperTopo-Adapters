# src/train.py
import argparse, os, json
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import make_tokenizer, SpatialDecoder, SimpleDecoder
from src.losses.hyperbolic import info_nce_from_pairs
from src.losses.segmentation import combo_loss
from src.utils.training import evaluate

# ---------------- Topology imports: robust fallback chain ----------------
_HAS_TOPO = False
topology_surrogate_loss = None
euler_characteristic_loss = None
try:
    # Preferred: EC + TV combined surrogate
    from src.losses.topology import topology_surrogate_loss, euler_characteristic_loss
    _HAS_TOPO = True
except Exception:
    # Try EC-only
    try:
        from src.losses.topology import euler_characteristic_loss
        _HAS_TOPO = True
    except Exception:
        _HAS_TOPO = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_ckpt', action='store_true', help='also save best-by-Dice head/decoder checkpoint')
    ap.add_argument('--topk', type=int, default=5, help='K for Top-K Dice selection (min-PD among Top-K)')
    return ap.parse_args()


def count_trainable(params):
    return sum(p.numel() for p in params if p.requires_grad)


def main():
    args = parse_args()
    cfg = load_config(args.config).raw
    device = args.device

    out_dir = cfg['logging']['out_dir']
    ensure_dir(out_dir)

    # ---------------- Data ----------------
    dcfg = cfg['data']
    patch_hw = tuple(dcfg['patch_size'])
    train_ds = LesionDataset(dcfg['train_csv'], dcfg['image_key'], dcfg['mask_key'], patch_hw)
    val_ds   = LesionDataset(dcfg['val_csv'],   dcfg['image_key'], dcfg['mask_key'], patch_hw)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['optimizer']['batch_size'],
        shuffle=True, num_workers=dcfg['num_workers']
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['optimizer']['batch_size'],
        shuffle=False, num_workers=dcfg['num_workers']
    )

    # ---------------- Model ----------------
    mcfg = cfg['model']
    token_dim     = int(mcfg.get('token_dim', 256))
    backbone_name = mcfg.get('backbone', 'tiny_tokenizer')
    train_tok     = mcfg.get('train_tokenizer', False)        # False | True | "partial"
    unfreeze_keys = mcfg.get('unfreeze_keys', [])             # list of substrings
    eu_only       = bool(mcfg.get('euclidean_only', False))

    tokenizer = make_tokenizer(
        backbone_name=backbone_name,
        de=token_dim,
        train_tokenizer=train_tok,
        unfreeze_keys=unfreeze_keys
    ).to(device)

    # Probe low-res grid size for decoder
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *patch_hw, device=device)
        _, out_hw = tokenizer(dummy)  # (Hp, Wp)

    head = ProductManifoldHead(
        in_dim=token_dim,
        dh=mcfg['latent_dims']['dh'],
        de=mcfg['latent_dims']['de'],
        ds=mcfg['latent_dims']['ds'],
        learn_curvature=mcfg['curvature']['learn'],
        euclidean_only=eu_only
    ).to(device)

    lat_dims = mcfg['latent_dims']
    tangent_dim = lat_dims['de'] if eu_only else (lat_dims['de'] + lat_dims['dh'] + lat_dims['ds'])
    dec_in_dim = token_dim + tangent_dim
    decoder = SpatialDecoder(in_dim=dec_in_dim, low_hw=out_hw).to(device)
    # decoder = SimpleDecoder(in_dim=dec_in_dim, out_hw=out_hw).to(device)  # optional ablation

    # ---------------- Optimizer ----------------
    base_lr = float(cfg['optimizer']['lr'])
    bb_lr   = float(cfg['optimizer'].get('backbone_lr', base_lr * 0.1))

    head_dec_params = list(head.parameters()) + list(decoder.parameters())
    backbone_params = [p for p in tokenizer.parameters() if p.requires_grad]

    opt_groups = [{"params": head_dec_params, "lr": base_lr}]
    if len(backbone_params) > 0:
        opt_groups.append({"params": backbone_params, "lr": bb_lr})

    opt = torch.optim.AdamW(opt_groups, weight_decay=cfg['optimizer']['weight_decay'])

    if args.dry_run:
        n_head_dec = count_trainable(head_dec_params)
        n_backbone = count_trainable(backbone_params)
        print("Dry run successful: data+model+optimizer instantiated.")
        print(f"[INFO] Backbone: {backbone_name} | token_dim={token_dim} | out_hw={out_hw}")
        print(f"[INFO] train_tokenizer={train_tok} | unfreeze_keys={unfreeze_keys}")
        print(f"[INFO] LR head/dec={base_lr} | LR backbone={bb_lr} | params head/dec={n_head_dec} | backbone trainable={n_backbone}")
        print(f"[INFO] Topology modules available: {_HAS_TOPO} "
              f"(surrogate={'yes' if topology_surrogate_loss is not None else 'no'}, "
              f"EC={'yes' if euler_characteristic_loss is not None else 'no'})")
        return

    # ---------------- Loss weights & warm-ups ----------------
    loss_cfg = cfg['loss']
    w_dice = float(loss_cfg.get('dice_weight', 1.0))
    w_bce  = float(loss_cfg.get('bce_weight', 0.5))
    w_hyp  = float(loss_cfg.get('hyperbolic_metric_weight', 0.25))
    w_topo = float(loss_cfg.get('topology_weight', 0.0))
    temp   = float(loss_cfg.get('contrastive', {}).get('temperature', 0.1))

    hyp_warm  = int(loss_cfg.get('hyp_warmup_epochs', 0))
    topo_warm = int(loss_cfg.get('topo_warmup_epochs', 0))

    # YAML-controlled topo_surrogate:
    raw_topo_cfg = loss_cfg.get('topo_surrogate', None)   # dict | false | None
    use_topo_surrogate = (raw_topo_cfg is not False) and isinstance(raw_topo_cfg, dict)

    if use_topo_surrogate:
        topo_thresholds = tuple(raw_topo_cfg.get('thresholds', [0.3, 0.5, 0.7]))
        topo_sigma      = float(raw_topo_cfg.get('sigma', 0.1))
        topo_alpha      = float(raw_topo_cfg.get('alpha', 1.0e-3))
        topo_beta       = float(raw_topo_cfg.get('beta',  1.0e-3))
    else:
        topo_thresholds = (0.5,)
        topo_sigma      = 0.1
        topo_alpha      = 0.0
        topo_beta       = 0.0

    best = {"dice": 0.0}
    epochs = int(cfg['optimizer']['epochs'])

    # ---------- keep Top-K by Dice and pick min-PD ----------
    TOPK = int(args.topk)
    topk = []  # list of dicts: {epoch, dice, pd, state}

    for epoch in range(epochs):
        tokenizer.train(bool(train_tok))
        head.train(); decoder.train()

        # scales are epoch-wise (warm-ups)
        hyp_scale  = w_hyp  * (min(1.0, (epoch + 1) / max(hyp_warm, 1))  if hyp_warm  > 0 else w_hyp  / max(w_hyp, 1.0))
        topo_scale = w_topo * (min(1.0, (epoch + 1) / max(topo_warm, 1)) if topo_warm > 0 else w_topo / max(w_topo, 1.0)) if w_topo > 0 else 0.0

        for batch in train_loader:
            x = batch['image'].to(device)
            y = batch['mask'].to(device)

            tokens, _ = tokenizer(x)
            lat = head(tokens)
            pooled = lat['tangent'].mean(dim=1)
            pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)

            pred = decoder(pooled_in, upsample_to=patch_hw)  # [B,1,H,W]

            # Segmentation
            loss_seg = combo_loss(pred, y, w_dice, w_bce)

            # Hyperbolic metric term
            loss_hyp = torch.tensor(0.0, device=device)
            if (not eu_only) and (w_hyp > 0.0) and (lat.get('latent_H', None) is not None):
                B, T, _ = lat['latent_H'].shape
                z = lat['latent_H'].reshape(B * T, -1)
                batch_ids = torch.arange(B, device=z.device).repeat_interleave(T)
                loss_hyp = info_nce_from_pairs(z, batch_ids, temperature=temp) * hyp_scale

            # Topology term (surrogate → EC-only → PD)
            loss_topo = torch.tensor(0.0, device=device)
            if w_topo > 0.0 and topo_scale > 0.0:
                if _HAS_TOPO:
                    if topology_surrogate_loss is not None and use_topo_surrogate:
                        loss_topo = topology_surrogate_loss(
                            pred, y,
                            thresholds=topo_thresholds, sigma=topo_sigma,
                            alpha=topo_alpha, beta=topo_beta,
                            normalize=True, reduction='mean'
                        ) * topo_scale
                    elif euler_characteristic_loss is not None:
                        loss_topo = euler_characteristic_loss(
                            pred, y,
                            thresholds=topo_thresholds, sigma=topo_sigma,
                            normalize=True, reduction='mean'
                        ) * topo_scale
                    else:
                        # Should not hit (since _HAS_TOPO False if neither importable)
                        loss_topo = torch.tensor(0.0, device=device)
                else:
                    # Fallback: PD (non-diff) — last resort
                    from src.utils.metrics import pd_distance_batch
                    pd_dist = pd_distance_batch(pred.detach(), y.detach())
                    if pd_dist == pd_dist:  # not NaN
                        loss_topo = torch.tensor(pd_dist, device=device) * topo_scale

            # Total
            loss = loss_seg + loss_hyp + loss_topo
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(decoder.parameters()), max_norm=1.0)
            opt.step()

        # ---------------- Validation ----------------
        metrics = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder},
                           val_loader, device=device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"VAL Dice={metrics['dice']:.3f} IoU={metrics['iou']:.3f} "
            f"BF1={metrics['boundary_f1']:.3f} Δβ0={metrics['betti0_err']:.2f} "
            f"Δβ1={metrics['betti1_err']:.2f} PD={metrics['pd_dist']:.3f} | "
            f"geom={'E-only' if eu_only else 'H⊕E⊕S'} topo_w={w_topo} "
            f"| scales: hyp={hyp_scale:.3f} topo={topo_scale:.3f} "
            f"| topo_mode={'surrogate' if (use_topo_surrogate and topology_surrogate_loss is not None and _HAS_TOPO) else ('EC-only' if (_HAS_TOPO and euler_characteristic_loss is not None) else ('PD' if w_topo>0 else 'off'))}"
        )

        # Save Top-K by Dice (store PD & state)
        cand = {
            "epoch": epoch + 1,
            "dice":  float(metrics['dice']),
            "pd":    float(metrics['pd_dist']),
            "state": {
                "head":    head.state_dict(),
                "decoder": decoder.state_dict(),
                **({"tokenizer": tokenizer.state_dict()} if mcfg.get('train_tokenizer', False) else {})
            }
        }
        topk.append(cand)
        topk.sort(key=lambda r: r["dice"], reverse=True)
        topk = topk[:TOPK]

        # Best-by-Dice (reference / optional ckpt)
        if metrics['dice'] > (best.get("dice", 0.0) or 0.0):
            best = metrics.copy()
            with open(os.path.join(out_dir, 'best_metrics.json'), 'w') as f:
                json.dump(best, f, indent=2)
            if args.save_ckpt:
                torch.save(cand["state"], os.path.join(out_dir, "best_adapter.pt"))

    # ---------------- Selection: min-PD within Top-K Dice ----------------
    chosen = min(topk, key=lambda r: r["pd"]) if len(topk) else None
    if chosen:
        torch.save(chosen["state"], os.path.join(out_dir, "ckpt_minPD_in_topK.pt"))
        with open(os.path.join(out_dir, "selection_minPD_in_topK.json"), "w") as f:
            json.dump({
                "topK": [{"epoch": r["epoch"], "dice": r["dice"], "pd": r["pd"]} for r in topk],
                "chosen": {"epoch": chosen["epoch"], "dice": chosen["dice"], "pd": chosen["pd"]}
            }, f, indent=2)
        print(f"[SELECT] min-PD in top-{TOPK}: epoch={chosen['epoch']} Dice={chosen['dice']:.4f} PD={chosen['pd']:.4f}")
    else:
        print("[SELECT] no candidates in Top-K selection (empty Top-K list).")

    print("Training complete. Best-by-Dice (reference):", best)


if __name__ == '__main__':
    main()
