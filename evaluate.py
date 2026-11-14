"""Compute FID and Inception Score with PyTorch-Ignite, following
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
(generated and real images are resized to 299x299 and fed to InceptionV3).

    python evaluate.py --ckpt runs/dcgan64/last.pt --data data/wikiart_64.npy
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from models import build_models


class UInt8Images(Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.arr[i])).permute(2, 0, 1)


def to_inception(x):
    """[-1,1] images -> [0,1], 299x299 (as in the Ignite blog's `interpolate`)."""
    x = ((x + 1) / 2).clamp(0, 1)
    return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=None, help="preprocessed .npy (default: the one used for training)")
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--real_is", action="store_true", help="also report IS of real images (reference)")
    p.add_argument("--n_individual", type=int, default=0, help="also save N single generated images")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ckpt["args"])
    G, _ = build_models(targs)
    G.load_state_dict(ckpt["G"])
    G = G.to(device).eval()
    out_dir = os.path.join(os.path.dirname(a.ckpt), f"eval_{os.path.splitext(os.path.basename(a.ckpt))[0]}")
    os.makedirs(out_dir, exist_ok=True)

    # real images: random subset of the training set
    arr = np.load(a.data or targs.data, mmap_mode="r")
    n = min(a.n_samples, len(arr))
    idx = np.sort(np.random.choice(len(arr), n, replace=False))
    loader = DataLoader(UInt8Images(arr[idx]), batch_size=a.batch_size, shuffle=False, num_workers=2)

    def evaluation_step(engine, batch):
        with torch.no_grad():
            real = batch.to(device).float().div(127.5).sub(1)
            noise = torch.randn(real.size(0), targs.nz, 1, 1, device=device)
            fake = G(noise)
            return to_inception(fake), to_inception(real)

    evaluator = Engine(evaluation_step)
    FID(device=device).attach(evaluator, "fid")
    InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")
    if a.real_is:
        InceptionScore(device=device, output_transform=lambda x: x[1]).attach(evaluator, "is_real")

    print(f"Evaluating {n} generated vs {n} real images on {device} ...")
    evaluator.run(loader, max_epochs=1)
    m = evaluator.state.metrics
    results = {"ckpt": a.ckpt, "epoch": ckpt["epoch"], "n_samples": n,
               "FID": float(m["fid"]), "IS": float(m["is"])}
    if a.real_is:
        results["IS_real"] = float(m["is_real"])
    print(json.dumps(results, indent=2))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # example images for the report
    with torch.no_grad():
        fake = G(torch.randn(64, targs.nz, 1, 1, device=device))
    save_image(fake, os.path.join(out_dir, "generated_grid.png"), nrow=8, normalize=True, value_range=(-1, 1))
    real = torch.from_numpy(np.ascontiguousarray(arr[idx[:64]])).permute(0, 3, 1, 2).float() / 255
    save_image(real, os.path.join(out_dir, "real_grid.png"), nrow=8)
    if a.n_individual:
        ind_dir = os.path.join(out_dir, "individual")
        os.makedirs(ind_dir, exist_ok=True)
        with torch.no_grad():
            imgs = G(torch.randn(a.n_individual, targs.nz, 1, 1, device=device))
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(ind_dir, f"{i:04d}.png"), normalize=True, value_range=(-1, 1))
    print(f"Saved metrics and image grids to {out_dir}")


if __name__ == "__main__":
    main()
