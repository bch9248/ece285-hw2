"""Train a DCGAN on preprocessed WikiArt.

    python train.py --data data/wikiart_64.npy --out runs/dcgan64 --epochs 30
    python train.py --resume runs/dcgan64/last.pt          # continue a run
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models import build_models
from plot_losses import plot


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/wikiart_64.npy")
    p.add_argument("--out", default="runs/dcgan64")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--nz", type=int, default=100, help="latent dim")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--loss", choices=["bce", "hinge"], default="bce")
    p.add_argument("--real_label", type=float, default=0.9, help="one-sided label smoothing (bce only)")
    p.add_argument("--sn", action="store_true", help="spectral norm in D (replaces BatchNorm)")
    p.add_argument("--no_flip", action="store_true", help="disable random horizontal flips")
    p.add_argument("--data_on_cpu", action="store_true", help="keep dataset in CPU RAM instead of GPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=5, help="keep a checkpoint every N epochs")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    return p.parse_args()


def d_loss_fn(args, d_real, d_fake):
    if args.loss == "hinge":
        return F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
    return (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, args.real_label))
            + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))


def g_loss_fn(args, d_fake):
    if args.loss == "hinge":
        return -d_fake.mean()
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def main():
    args = get_args()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        epochs = args.epochs
        args = argparse.Namespace(**ckpt["args"])
        args.epochs = max(epochs, args.epochs)  # allow extending a run
        args.resume = None

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    # ---- data: uint8 (N, H, W, 3) -> uint8 (N, 3, H, W), kept in memory ----
    arr = np.load(args.data)
    data = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    args.img_size = data.shape[-1]
    if device.type == "cuda" and not args.data_on_cpu:
        data = data.to(device)
    elif device.type == "cuda":
        data = data.pin_memory()
    N = data.shape[0]
    steps_per_epoch = N // args.batch_size
    print(f"Data: {N} images of {args.img_size}x{args.img_size} on {data.device}; "
          f"{steps_per_epoch} steps/epoch; device={device}")

    # ---- models ----
    G, D = build_models(args)
    G, D = G.to(device), D.to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    start_epoch, step = 0, 0
    if ckpt is not None:
        G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"]); optD.load_state_dict(ckpt["optD"])
        fixed_noise = ckpt["fixed_noise"].to(device)
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"Resumed from epoch {ckpt['epoch']} (step {step})")
    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M | "
          f"D params: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")

    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)  # hyperparameters for the report

    csv_path = os.path.join(args.out, "losses.csv")
    new_csv = ckpt is None or not os.path.exists(csv_path)
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["epoch", "step", "loss_D", "loss_G", "D_x", "D_G_z"])

    def to_float(x):  # uint8 [0,255] -> float [-1,1], with random h-flip
        x = x.to(device, non_blocking=True).float().div_(127.5).sub_(1)
        if not args.no_flip:
            flip = torch.rand(x.size(0), device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x

    def save_ckpt(path, epoch):
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(),
                    "optD": optD.state_dict(), "epoch": epoch, "step": step,
                    "fixed_noise": fixed_noise.cpu(), "args": vars(args)}, path)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        perm = torch.randperm(N, device=data.device)
        for i in range(steps_per_epoch):
            real = to_float(data[perm[i * args.batch_size:(i + 1) * args.batch_size]])
            z = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = G(z)

            # --- update D ---
            d_real, d_fake = D(real), D(fake.detach())
            loss_D = d_loss_fn(args, d_real, d_fake)
            optD.zero_grad(set_to_none=True)
            loss_D.backward()
            optD.step()

            # --- update G ---
            d_fake_g = D(fake)
            loss_G = g_loss_fn(args, d_fake_g)
            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            with torch.no_grad():
                D_x = torch.sigmoid(d_real).mean().item()
                D_G_z = torch.sigmoid(d_fake).mean().item()
            writer.writerow([epoch, step, f"{loss_D.item():.5f}", f"{loss_G.item():.5f}",
                             f"{D_x:.4f}", f"{D_G_z:.4f}"])
            if step % args.log_every == 0:
                csv_f.flush()
                print(f"[ep {epoch:03d} {i:5d}/{steps_per_epoch}] loss_D {loss_D.item():.3f} "
                      f"loss_G {loss_G.item():.3f} D(x) {D_x:.3f} D(G(z)) {D_G_z:.3f} "
                      f"| {(time.time() - t0) / 60:.1f} min", flush=True)
            step += 1

        # --- end of epoch: samples + checkpoints ---
        G.eval()
        with torch.no_grad():
            save_image(G(fixed_noise), os.path.join(args.out, "samples", f"epoch_{epoch:03d}.png"),
                       nrow=8, normalize=True, value_range=(-1, 1))
        save_ckpt(os.path.join(args.out, "last.pt"), epoch)
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_ckpt(os.path.join(args.out, f"epoch_{epoch:03d}.pt"), epoch)
        csv_f.flush()

    csv_f.close()
    plot(csv_path, os.path.join(args.out, "loss_curves"))
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs in {args.out}")


if __name__ == "__main__":
    main()
