#OLD EVAL FILE 
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder
from src.utils.training import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = load_config(args.config).raw; device = args.device
    out_dir = cfg['logging']['out_dir']; ensure_dir(out_dir)
    dcfg = cfg['data']
    val_ds = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    test_ds= LesionDataset(dcfg['test_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    val_loader = DataLoader(val_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])
    test_loader= DataLoader(test_ds, batch_size=cfg['optimizer']['batch_size'], shuffle=False, num_workers=dcfg['num_workers'])
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
    metrics_val = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, val_loader, device=device)
    metrics_test = evaluate({"tokenizer": tokenizer, "head": head, "decoder": decoder}, test_loader, device=device)
    with open(os.path.join(out_dir, 'metrics_val.json'), 'w') as f: json.dump(metrics_val, f, indent=2)
    with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f: json.dump(metrics_test, f, indent=2)
    print('VAL:', metrics_val); print('TEST:', metrics_test)

if __name__ == '__main__':
    main()
