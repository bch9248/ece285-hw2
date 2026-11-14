"""Resize the raw WikiArt images once into a compact uint8 .npy array.

The raw dataset is ~80k large JPEGs; decoding them every epoch is very slow.
Preprocessing once makes training GPU-bound instead of disk-bound.

Example:
    python prepare_data.py --src /path/to/public/wikiart --size 64 --out data/wikiart_64.npy
    python prepare_data.py --src /path/to/public/wikiart --size 64 --styles Impressionism Post_Impressionism
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # some WikiArt paintings are huge
ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root, styles=None):
    styles = set(styles) if styles else None
    paths = []
    for dirpath, _, files in os.walk(root):
        if styles is not None and os.path.basename(dirpath) not in styles:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths


def load_one(job):
    path, size = job
    try:
        with Image.open(path) as img:
            img.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode
            img = img.convert("RGB")
            w, h = img.size
            s = size / min(w, h)
            nw, nh = max(size, round(w * s)), max(size, round(h * s))
            img = img.resize((nw, nh), Image.BICUBIC)
            left, top = (nw - size) // 2, (nh - size) // 2
            img = img.crop((left, top, left + size, top + size))  # center crop
            return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="unzipped WikiArt root (style sub-folders)")
    p.add_argument("--out", default=None, help="output .npy (default data/wikiart_<size>.npy)")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--styles", nargs="*", default=None, help="only these style folders")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = p.parse_args()

    out = args.out or f"data/wikiart_{args.size}.npy"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    paths = find_images(args.src, args.styles)
    if args.max_images:
        rng = np.random.default_rng(0)
        paths = sorted(rng.choice(paths, size=min(args.max_images, len(paths)), replace=False))
    if not paths:
        raise SystemExit(f"No images found under {args.src} (styles={args.styles})")
    print(f"Found {len(paths)} images, resizing to {args.size}x{args.size} with {args.workers} workers")

    arr = np.empty((len(paths), args.size, args.size, 3), dtype=np.uint8)
    n_ok = 0
    with Pool(args.workers) as pool:
        jobs = ((pth, args.size) for pth in paths)
        for img in tqdm(pool.imap(load_one, jobs, chunksize=64), total=len(paths)):
            if img is not None:
                arr[n_ok] = img
                n_ok += 1
    arr = arr[:n_ok]
    np.save(out, arr)
    meta = {"src": os.path.abspath(args.src), "size": args.size, "styles": args.styles,
            "n_found": len(paths), "n_saved": int(n_ok), "n_failed": len(paths) - n_ok}
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Saved {out}  shape={arr.shape}  ({arr.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
