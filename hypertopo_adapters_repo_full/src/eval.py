# src/eval.py
import argparse, json, os
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import make_tokenizer, SpatialDecoder, SimpleDecoder
from src.utils.training import evaluate


def load_partial_state(module: torch.nn.Module, state_dict: dict | None, tag: str):
    """
    Safely load only matching keys (so eval doesn't break if the tokenizer wasn't saved, etc.).
    """
    if not state_dict:
        print(f"[INFO] No state for {tag} (skipping).")
        return
    msg = module.load_state_dict(state_dict, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    if missing or unexpected:
        print(f"[WARN] Partial load for {tag}: missing={missing[:6]} unexpected={unexpected[:6]}")


def choose_ckpt(out_dir: str, arg_ckpt: str | None, prefer_minpd: bool) -> str | None:
    """
    Resolve which checkpoint to load:
      - if --ckpt is a path, return it (if exists)
      - else if prefer_minpd and ckpt_minPD_in_topK.pt exists, use it
      - else if best_adapter.pt exists, use it
      - else None
    """
    if arg_ckpt:
        if os.path.isfile(arg_ckpt):
            return arg_ckpt
        print(f"[WARN] --ckpt path not found: {arg_ckpt}")
        return None
    if prefer_minpd:
        p = os.path.join(out_dir, "ckpt_minPD_in_topK.pt")
        if os.path.isfile(p):
            return p
    p = os.path.join(out_dir, "best_adapter.pt")
    if os.path.isfile(p):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    ap.add_argument('--ckpt', default=None,
                    help='Checkpoint path. If omitted, tries ckpt_minPD_in_topK.pt then best_adapter.pt.')
    ap.add_argument('--prefer_minpd', action='store_true',
                    help='Prefer ckpt_minPD_in_topK.pt over best_adapter.pt when --ckpt not given.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--use_simple_decoder', action='store_true',
                    help='Use SimpleDecoder instead of SpatialDecoder (for ablations).')
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)

    # ---------------- Data ----------------
    dcfg = cfg['data']
    patch_hw = tuple(dcfg['patch_size'])
    val_ds  = LesionDataset(dcfg['val_csv'],  dcfg['image_key'], dcfg['mask_key'], patch_hw)
    test_ds = LesionDataset(dcfg['test_csv'], dcfg['image_key'], dcfg['mask_key'], patch_hw)

    val_loader  = DataLoader(val_ds,  batch_size=cfg['optimizer']['batch_size'],
                             shuffle=False, num_workers=dcfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=cfg['optimizer']['batch_size'],
                             shuffle=False, num_workers=dcfg['num_workers'])

    # ---------------- Model (tokenizer/head/decoder) ----------------
    mcfg = cfg['model']
    token_dim     = int(mcfg.get('token_dim', 256))
    backbone_name = mcfg.get('backbone', 'tiny_tokenizer')
    eu_only       = bool(mcfg.get('euclidean_only', False))
    train_tok     = mcfg.get('train_tokenizer', False)          # False | True | "partial"
    unfreeze_keys = mcfg.get('unfreeze_keys', [])

    # tokenizer/backbone (frozen/partial/full) — safe in eval; we just won't train it
    tokenizer = make_tokenizer(
        backbone_name=backbone_name,
        de=token_dim,
        train_tokenizer=train_tok,
        unfreeze_keys=unfreeze_keys
    ).to(device)

    # Probe low-res grid size for decoder
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *patch_hw, device=device)
        _, out_hw = tokenizer(dummy)            # (H', W') depends on backbone stride/patch

    head = ProductManifoldHead(
        in_dim=token_dim,
        dh=mcfg['latent_dims']['dh'],
        de=mcfg['latent_dims']['de'],
        ds=mcfg['latent_dims']['ds'],
        learn_curvature=mcfg['curvature']['learn'],
        euclidean_only=eu_only
    ).to(device)

    lat = mcfg['latent_dims']
    tangent_dim = lat['de'] if eu_only else (lat['de'] + lat['dh'] + lat['ds'])
    dec_in_dim = token_dim + tangent_dim

    if args.use_simple_decoder:
        decoder = SimpleDecoder(in_dim=dec_in_dim, out_hw=out_hw).to(device)
    else:
        decoder = SpatialDecoder(in_dim=dec_in_dim, low_hw=out_hw).to(device)

    # ---------------- Optional checkpoint load ----------------
    ckpt_path = choose_ckpt(out_dir, args.ckpt, args.prefer_minpd)
    if ckpt_path:
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        load_partial_state(head,    ckpt.get('head'),    'head')
        load_partial_state(decoder, ckpt.get('decoder'), 'decoder')
        # Load tokenizer only if saved (e.g., when train_tokenizer was True/partial during training)
        if 'tokenizer' in ckpt:
            load_partial_state(tokenizer, ckpt.get('tokenizer'), 'tokenizer')
    else:
        print("[INFO] No checkpoint loaded (random head/decoder weights).")

    # ---------------- Evaluate ----------------
    metrics_val  = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder},
                            val_loader, device=device)
    metrics_test = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder},
                            test_loader, device=device)

    with open(os.path.join(out_dir, 'metrics_val.json'), 'w') as f:
        json.dump(metrics_val, f, indent=2)
    with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f:
        json.dump(metrics_test, f, indent=2)

    geom = 'E-only' if eu_only else 'H⊕E⊕S'
    print(f"[SUMMARY] backbone={backbone_name} geom={geom} out_hw={out_hw} "
          f"train_tok={train_tok} unfreeze_keys={unfreeze_keys}")
    print('VAL:', metrics_val)
    print('TEST:', metrics_test)


if __name__ == '__main__':
    main()
