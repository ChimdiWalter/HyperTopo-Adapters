import argparse, csv, os, glob, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root with images/ and masks/')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--suffix_img', default='.png')
    ap.add_argument('--suffix_mask', default='.png')
    ap.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    args = ap.parse_args()
    root = Path(args.root)
    imgs = sorted(glob.glob(str(root / 'images' / f'*{args.suffix_img}')))
    rows = []
    for img in imgs:
        stem = Path(img).stem
        mask = str(root / 'masks' / f'{stem}{args.suffix_mask}')
        if os.path.exists(mask):
            rows.append({'image_path': img, 'mask_path': mask})
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n*args.split[0]); n_val = int(n*args.split[1])
    splits = [('train.csv', rows[:n_train]), ('val.csv', rows[n_train:n_train+n_val]), ('test.csv', rows[n_train+n_val:])]
    os.makedirs(args.outdir, exist_ok=True)
    for name, data in splits:
        with open(Path(args.outdir)/name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image_path','mask_path'])
            w.writeheader(); w.writerows(data)
        print('Wrote', name, len(data), 'rows')

if __name__ == '__main__':
    main()
