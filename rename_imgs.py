"""
import os
from pathlib import Path


def rename_pngs(root, subdir):
    folder = Path(root) / subdir
    files = sorted(folder.glob("*.png"))
    for i, f in enumerate(files, start=1):
        new_name = f"IMG_{i:04d}.png"
        new_path = folder / new_name
        print(f"Renaming {f.name} -> {new_name}")
        f.rename(new_path)

if __name__ == "__main__":
    DATA_ROOT ="/cluster/VAST/kazict-lab/e/lesion_phes/code/dataset_segmentation/Manifold_experiments/data_root"
    rename_pngs(DATA_ROOT, "images")
    rename_pngs(DATA_ROOT, "masks")
    """


#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps
import argparse, shutil, re

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def idx(folder: Path):
    return {p.stem: p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS}

def save_image_png(src: Path, dst: Path):
    im = Image.open(src).convert("RGB")
    im = ImageOps.exif_transpose(im)  # fix rotated JPEGs
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="PNG", compress_level=3)

def save_mask_png(src: Path, dst: Path, thresh=127):
    mk = Image.open(src).convert("L")
    # binarize to protect topology/boundary metrics
    mk = mk.point(lambda x: 255 if x > thresh else 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mk.save(dst, format="PNG", compress_level=0)

def main(root: Path, images_dir="Images", masks_dir="masks", commit=False, thresh=127):
    img_dir = root / images_dir
    msk_dir = root / masks_dir
    if not img_dir.exists() or not msk_dir.exists():
        raise SystemExit(f"[ERR] Missing {img_dir} or {msk_dir}")

    imgs, msks = idx(img_dir), idx(msk_dir)
    common = sorted(set(imgs) & set(msks))
    miss_i = sorted(set(imgs) - set(msks))
    miss_m = sorted(set(msks) - set(imgs))

    if miss_i:
        print(f"[WARN] {len(miss_i)} images have no mask (first 5): {miss_i[:5]}")
    if miss_m:
        print(f"[WARN] {len(miss_m)} masks have no image (first 5): {miss_m[:5]}")

    # Refuse to overwrite if IMG_####.png already exists
    pattern = re.compile(r'^IMG_\d{4}\.png$')
    collisions = [p for p in list(img_dir.glob('IMG_*.png')) + list(msk_dir.glob('IMG_*.png'))
                  if pattern.match(p.name)]
    if collisions and commit:
        raise SystemExit(f"[ERR] Found existing IMG_####.png files ({len(collisions)}). "
                         "Move them or run without --commit to preview.")

    plan = []
    for i, stem in enumerate(common, start=1):
        img_src, msk_src = imgs[stem], msks[stem]
        img_tmp = img_dir / f".tmp_IMG_{i:04d}.png"
        msk_tmp = msk_dir / f".tmp_IMG_{i:04d}.png"
        plan.append((img_src, img_tmp, msk_src, msk_tmp))

    print(f"[INFO] Paired files to process: {len(plan)}")
    if not commit:
        for a, _, c, _ in plan[:10]:
            print(f"  - {a.name} + {c.name} -> IMG_####.png")
        print("[DRY-RUN] No files written. Re-run with --commit to apply.")
        return

    # 1) Write converted PNGs to temporary names (no overwrite risk)
    for img_src, img_tmp, msk_src, msk_tmp in plan:
        save_image_png(img_src, img_tmp)
        save_mask_png(msk_src, msk_tmp, thresh=thresh)

    # 2) Backup originals
    b_i = img_dir / "_backup_originals"; b_i.mkdir(exist_ok=True)
    b_m = msk_dir / "_backup_originals"; b_m.mkdir(exist_ok=True)
    for img_src, img_tmp, msk_src, msk_tmp in plan:
        shutil.move(str(img_src), str(b_i / img_src.name))
        shutil.move(str(msk_src), str(b_m / msk_src.name))

    # 3) Promote temps to final names
    for _, img_tmp, __, msk_tmp in plan:
        img_final = img_dir / img_tmp.name.replace(".tmp_", "")
        msk_final = msk_dir / msk_tmp.name.replace(".tmp_", "")
        img_tmp.rename(img_final)
        msk_tmp.rename(msk_final)

    print("[DONE] Converted to PNG and renamed in place. Originals in _backup_originals/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data_root (contains Images/ and masks/)")
    ap.add_argument("--commit", action="store_true", help="Apply changes (omit for dry run)")
    ap.add_argument("--thresh", type=int, default=127, help="Binarization threshold for masks")
    args = ap.parse_args()
    main(Path(args.root), commit=args.commit, thresh=args.thresh)
