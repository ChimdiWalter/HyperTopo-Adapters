# diagnose_dataset.py  (FIXED)
import argparse, csv, os
from pathlib import Path

import numpy as np                     # <-- import at top (global)
from PIL import Image, ImageOps
from skimage.measure import label as cc_label

def load_csv(csv_path):
    rows=[]
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append((row['image_path'], row['mask_path']))
    return rows

def resize_like_train(img_pil, mask_pil, patch_h, patch_w):
    # image: bilinear; mask: nearest; binarize to {0,1}
    img = img_pil.convert('RGB').resize((patch_w, patch_h), Image.BILINEAR)
    m   = mask_pil.convert('L').resize((patch_w, patch_h), Image.NEAREST)
    m_np = np.array(m, dtype=np.uint8)
    if m_np.max() > 1:
        m_np = (m_np >= 128).astype(np.uint8)
    return img, m_np

def overlay(img, mask_np, alpha=0.5):
    mask_img = Image.fromarray((mask_np*255).astype(np.uint8))
    mask_rgb = ImageOps.colorize(mask_img, black=(0,0,0), white=(255,0,0))
    return Image.blend(img, mask_rgb, alpha=alpha)

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter/union) if union>0 else float('nan')

def dice(a, b):
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return (2*inter/s) if s>0 else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='train/val/test csv to check')
    ap.add_argument('--patch', type=int, nargs=2, default=[256,256], help='H W (match config.data.patch_size)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--limit', type=int, default=24)
    args = ap.parse_args()
    H,W = args.patch

    pairs = load_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    bad_paths=0; nonbinary=0; empties=0; huge_comp=0
    fg_fracs=[]; betti0_list=[]; sample_count=0
    iou_self_list=[]; dice_self_list=[]

    for i,(ip,mp) in enumerate(pairs):
        ip, mp = Path(ip), Path(mp)
        if not ip.exists() or not mp.exists():
            bad_paths += 1
            print(f'[MISSING] {ip} or {mp}')
            continue

        img = Image.open(ip)
        mask = Image.open(mp)
        img_r, m = resize_like_train(img, mask, H, W)

        # binary check
        uniq = np.unique(m)
        if not set(uniq.tolist()).issubset({0,1}):
            nonbinary += 1

        # foreground fraction
        fg = m.mean()
        fg_fracs.append(fg)
        if fg == 0.0:
            empties += 1

        # components (β0)
        beta0 = cc_label(m, connectivity=1).max()
        betti0_list.append(beta0)
        if beta0 > 20:
            huge_comp += 1

        # mask-vs-mask “self” metrics (must be 1.0)
        iou_self_list.append(iou(m, m))
        dice_self_list.append(dice(m, m))

        # save a few overlays
        if sample_count < args.limit:
            ov = overlay(img_r, m, 0.35)
            stem = f'{i:04d}'
            ov.save(os.path.join(args.outdir, f'overlay_{stem}.png'))
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(args.outdir,f'mask_resized_{stem}.png'))
            img_r.save(os.path.join(args.outdir, f'image_resized_{stem}.png'))
            sample_count += 1

    print('--- SUMMARY ---')
    print(f'Pairs: {len(pairs)} | Missing paths: {bad_paths}')
    if fg_fracs:
        fg_fracs = np.array(fg_fracs); betti0_list = np.array(betti0_list)
        print(f'Foreground px% mean={fg_fracs.mean():.4f}  median={np.median(fg_fracs):.4f}  min={fg_fracs.min():.4f}  max={fg_fracs.max():.4f}')
        print(f'β0 (components) mean={betti0_list.mean():.2f} median={np.median(betti0_list):.2f} max={betti0_list.max():.0f}')
    print(f'Non-binary masks: {nonbinary} | Empty masks: {empties} | Very fragmented (β0>20): {huge_comp}')
    if iou_self_list:
        print(f'IoU(mask,mask) mean={np.mean(iou_self_list):.3f}  (should be 1.000)')
        print(f'Dice(mask,mask) mean={np.mean(dice_self_list):.3f} (should be 1.000)')
    print(f'Overlays saved to: {args.outdir}')

if __name__ == '__main__':
    main()
