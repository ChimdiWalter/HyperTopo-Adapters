#NEWEST-OLD EVAL.py
import argparse, json, os
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import make_tokenizer, SimpleDecoder
from src.utils.training import evaluate

def load_partial_state(module, state):
    """
    Load only matching keys. Safe when the tokenizer is frozen and not saved.
    """
    if state is None:
        return
    msg = module.load_state_dict(state, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    if missing or unexpected:
        print(f"[WARN] Partial load: missing={missing[:5]}... unexpected={unexpected[:5]}...")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    ap.add_argument('--ckpt', default=None, help='Optional checkpoint .pt to load')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)

    # ---------------- Data ----------------
    dcfg = cfg['data']
    patch_hw = tuple(dcfg['patch_size'])
    val_ds  = LesionDataset(dcfg['val_csv'],  dcfg['image_key'], dcfg['mask_key'], patch_hw)
    test_ds = LesionDataset(dcfg['test_csv'], dcfg['image_key'], dcfg['mask_key'], patch_hw)

    val_loader = DataLoader(val_ds,  batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])
    test_loader= DataLoader(test_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])

    # ---------------- Model (tokenizer/head/decoder) ----------------
    mcfg = cfg['model']
    token_dim = int(mcfg.get('token_dim', 256))
    backbone_name = mcfg.get('backbone', 'tiny_tokenizer')
    eu_only = bool(mcfg.get('euclidean_only', False))

    tokenizer = make_tokenizer(
        backbone_name=backbone_name,
        de=token_dim,
        train_tokenizer=mcfg.get('train_tokenizer', False)
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
    decoder = SimpleDecoder(in_dim=dec_in_dim, out_hw=out_hw).to(device)

    # ---------------- Optional checkpoint load ----------------
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"[INFO] Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        load_partial_state(head,    ckpt.get('head'))
        load_partial_state(decoder, ckpt.get('decoder'))
        # tokenizer is usually frozen; load only if present (and if you trained it)
        if mcfg.get('train_tokenizer', False) and ('tokenizer' in ckpt):
            load_partial_state(tokenizer, ckpt['tokenizer'])
    else:
        if args.ckpt:
            print(f"[WARN] Checkpoint not found: {args.ckpt} (continuing with randomly initialized head/decoder)")

    # ---------------- Evaluate ----------------
    metrics_val  = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder},
                            val_loader, device=device)
    metrics_test = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder},
                            test_loader, device=device)

    with open(os.path.join(out_dir, 'metrics_val.json'), 'w') as f:
        json.dump(metrics_val, f, indent=2)
    with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f:
        json.dump(metrics_test, f, indent=2)

    print('VAL:', metrics_val)
    print('TEST:', metrics_test)

if __name__ == '__main__':
    main()

