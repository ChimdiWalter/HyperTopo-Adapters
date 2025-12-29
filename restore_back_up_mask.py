#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import argparse

def main(root):
    masks_dir = Path(root) / "masks"
    backup_dir = masks_dir / "_backup_originals"
    assert masks_dir.exists(), f"Missing {masks_dir}"
    assert backup_dir.exists(), f"Missing {backup_dir}"

    # current standardized targets
    targets = sorted(p for p in masks_dir.glob("IMG_*.png") if p.is_file())
    # backup originals (sorted by stem to match the original pairing order)
    backups = sorted((p for p in backup_dir.iterdir() if p.is_file()),
                     key=lambda x: x.stem.lower())

    if not targets:
        raise SystemExit("[ERR] No IMG_####.png masks found.")
    if not backups:
        raise SystemExit("[ERR] No backup originals found.")

    n = min(len(targets), len(backups))
    if len(targets) != len(backups):
        print(f"[WARN] counts differ: targets={len(targets)} backups={len(backups)}; restoring {n} pairs")

    # preview first few
    print("[INFO] Preview of mapping (first 5):")
    for i in range(min(5, n)):
        print(f"  {backups[i].name}  ->  {targets[i].name}")

    # restore: re-encode each backup as PNG into the standardized filename
    restored = 0
    for i in range(n):
        src = backups[i]
        dst = targets[i]
        with Image.open(src) as im:
            # Save as-is; no thresholding here
            im.save(dst, format="PNG")
        restored += 1

    print(f"[DONE] Restored {restored} masks from backups into {masks_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data_root (contains Images/ and masks/)")
    args = ap.parse_args()
    main(args.root)
