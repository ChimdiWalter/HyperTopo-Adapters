#OLD TRAIN.PY
import argparse, os, json
import torch
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder
from src.losses.hyperbolic import info_nce_from_pairs
from src.losses.segmentation import combo_loss
from src.utils.training import evaluate

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config).raw
    device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)

    # Data
    dcfg = cfg['data']
    train_ds = LesionDataset(dcfg['train_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    val_ds   = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    train_loader = DataLoader(train_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=True, num_workers=dcfg['num_workers'])
    val_loader   = DataLoader(val_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])

    # Model
    tokenizer = TinyTokenizer(out_dim=256).to(device)
    eu_only = cfg['model'].get('euclidean_only', False)
    head = ProductManifoldHead(in_dim=256,
                               dh=cfg['model']['latent_dims']['dh'],
                               de=cfg['model']['latent_dims']['de'],
                               ds=cfg['model']['latent_dims']['ds'],
                               learn_curvature=cfg['model']['curvature']['learn'],
                               euclidean_only=eu_only).to(device)
    in_dim = (256 + cfg['model']['latent_dims']['de']) if eu_only else (256 + sum(cfg['model']['latent_dims'].values()))
    decoder = SimpleDecoder(in_dim=in_dim, out_hw=(16,16)).to(device)

    params = list(head.parameters()) + list(decoder.parameters())
    if cfg['model'].get('train_tokenizer', False): params += list(tokenizer.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])

    if args.dry_run:
        print("Dry run successful: data+model+optimizer instantiated."); return

    w_dice, w_bce = cfg['loss']['dice_weight'], cfg['loss'].get('bce_weight', 0.5)
    w_hyp, w_topo = cfg['loss']['hyperbolic_metric_weight'], cfg['loss']['topology_weight']
    temp = cfg['loss']['contrastive']['temperature']

    best = {"dice": 0.0}
    for epoch in range(cfg['optimizer']['epochs']):
        tokenizer.train(False); head.train(); decoder.train()
        for batch in train_loader:
            x = batch['image'].to(device); y = batch['mask'].to(device)
            tokens, hw = tokenizer(x)
            lat = head(tokens)
            pooled = lat['tangent'].mean(dim=1)
            pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
            pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))

            # Segmentation loss
            loss_seg = combo_loss(pred, y, w_dice, w_bce)

            # Hyperbolic metric loss (skip if Euclidean-only)
            loss_hyp = torch.tensor(0.0, device=device)
            if (not eu_only) and (w_hyp > 0.0) and (lat['latent_H'] is not None):
                B, T, _ = lat['latent_H'].shape
                z = lat['latent_H'].reshape(B*T, -1)
                batch_ids = torch.arange(B, device=z.device).repeat_interleave(T)
                loss_hyp = info_nce_from_pairs(z, batch_ids, temperature=temp) * w_hyp

            # Topology term (optional; can be heavy)
            #from src.utils.metrics import pd_distance_batch
            #pd_dist = pd_distance_batch(pred.detach(), y.detach())
            #loss_topo = (torch.tensor(pd_dist, device=device) if pd_dist == pd_dist else torch.tensor(0.0, device=device)) * w_topo

            loss_topo = torch.tensor(0.0, device=device)
            if w_topo > 0.0:
                 from src.utils.metrics import pd_distance_batch
                 pd_dist = pd_distance_batch(pred.detach(), y.detach())
                 if pd_dist == pd_dist:  # not NaN
                    loss_topo = torch.tensor(pd_dist, device=device) * w_topo


            loss = loss_seg + loss_hyp + loss_topo
            opt.zero_grad(); loss.backward(); opt.step()

        metrics = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, val_loader, device=device)
        print(f"Epoch {epoch+1}/{cfg['optimizer']['epochs']} | val: {metrics}")
        if metrics['dice'] > best['dice']:
            best = metrics.copy()
            with open(os.path.join(out_dir, 'best_metrics.json'), 'w') as f: json.dump(best, f, indent=2)
    print("Training complete. Best val:", best)

if __name__ == '__main__':
    main()
