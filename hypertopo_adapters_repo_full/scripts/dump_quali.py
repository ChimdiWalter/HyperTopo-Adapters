import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils.io import load_config, ensure_dir
from src.data.lesion_dataset import LesionDataset
from src.models.manifold import ProductManifoldHead
from src.models.backbone import TinyTokenizer, SimpleDecoder

def save_triplet(img, gt, pred, path):
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(img.transpose(1,2,0)); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(gt[0], cmap='gray');   axs[1].set_title('GT'); axs[1].axis('off')
    axs[2].imshow(pred[0], cmap='gray'); axs[2].set_title('Pred'); axs[2].axis('off')
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num', type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    device = args.device
    out_dir = os.path.join(cfg['logging']['out_dir'], 'qualitative'); ensure_dir(out_dir)

    dcfg = cfg['data']
    val_ds = LesionDataset(dcfg['val_csv'], dcfg['image_key'], dcfg['mask_key'], tuple(dcfg['patch_size']))
    loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=dcfg['num_workers'])

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

    tokenizer.eval(); head.eval(); decoder.eval()
    count = 0
    for batch in loader:
        x, y = batch['image'].to(device), batch['mask'].to(device)
        tokens, hw = tokenizer(x)
        lat = head(tokens)
        pooled = lat['tangent'].mean(dim=1)
        pooled_in = torch.cat([tokens.mean(dim=1), pooled], dim=-1)
        pred = decoder(pooled_in, upsample_to=(x.shape[-2], x.shape[-1]))
        img_np  = x[0].detach().cpu().numpy()
        gt_np   = y[0].detach().cpu().numpy()
        pred_np = (pred[0].detach().cpu().numpy() > 0.5).astype(np.float32)
        save_triplet(img_np, gt_np, pred_np, os.path.join(out_dir, f'sample_{count:03d}.png'))
        count += 1
        if count >= args.num: break
    print('Saved', count, 'samples to', out_dir)

if __name__ == '__main__':
    main()
