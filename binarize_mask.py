#!/usr/bin/env python3
# binarize_masks_inplace.py
from pathlib import Path
from PIL import Image
import argparse, shutil, numpy as np

def binarize_array(arr: np.ndarray):
    # Dominant-background binarization
    vals, cnts = np.unique(arr, return_counts=True)
    bg = vals[np.argmax(cnts)]
    fg = (arr != bg)

    # Heuristic guardrail: if foreground covers ~everything, assume inversion
    fg_ratio = float(fg.mean())
    if fg_ratio > 0.95:
        # try treating bg as foreground instead (rare cases)
        fg = (arr == bg)
        fg_ratio = float(fg.mean())
    bin_arr = (fg.astype(np.uint8) * 255)
    return bin_arr, bg, fg_ratio

def main(root: Path, masks_dir: str, commit: bool):
    mdir = root / masks_dir
    assert mdir.exists(), f"[ERR] Missing masks dir: {mdir}"

    # process common mask formats; change if you need more
    files = sorted([p for p in mdir.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}])
    if not files:
        raise SystemExit(f"[ERR] No mask files found in {mdir}")

    backup = mdir / "_backup_before_binarize"
    tmpdir = mdir / "_tmp_binarize"
    flips = 0
    empties = 0

    print(f"[INFO] Found {len(files)} mask files in {mdir}")
    if not commit:
        print("[DRY-RUN] Previewing changes; no files will be modified.")

    # Prepare dirs only if committing
    if commit:
        backup.mkdir(exist_ok=True)
        tmpdir.mkdir(exist_ok=True)

    for p in files:
        # Load preserving labels if paletted/grayscale
        im = Image.open(p)
        if im.mode not in ("L", "P"):
            im = im.convert("L")
        arr = np.array(im)

        bin_arr, bg, fg_ratio = binarize_array(arr)
        if fg_ratio == 0.0:
            empties += 1
        if fg_ratio > 0.95:
            flips += 1  # inversion heuristic triggered

        if commit:
            # Write to temp, back up original, then move into place
            tmp_out = tmpdir / (p.stem + ".png")
            Image.fromarray(bin_arr, mode="L").save(tmp_out, format="PNG", compress_level=0)
            shutil.move(str(p), str(backup / p.name))   # backup original
            shutil.move(str(tmp_out), str(mdir / (p.stem + ".png")))  # final name (forces .png)

    print(f"[SUMMARY] processed={len(files)} | empties(after bin)= {empties} | flips(>95% fg)={flips}")
    if commit:
        print(f"[DONE] Originals moved to: {backup}")
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print("[DRY-RUN] Re-run with --commit to apply changes.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data_root (contains masks/)")
    ap.add_argument("--masks_dir", default="masks", help="Masks folder name (default: masks)")
    ap.add_argument("--commit", action="store_true", help="Actually overwrite files")
    args = ap.parse_args()
    main(Path(args.root), args.masks_dir, args.commit)
