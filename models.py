"""DCGAN generator / discriminator, sized by image resolution (64, 128, ...)."""
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def _init(m):
    # DCGAN init (Radford et al., 2015)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    return m


def _n_blocks(img_size):
    n = int(math.log2(img_size))
    assert 2 ** n == img_size and img_size >= 32, "img_size must be a power of 2 >= 32"
    return n - 3  # number of middle (up/down) blocks


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        super().__init__()
        n = _n_blocks(img_size)
        mult = 2 ** n
        layers = [
            _init(nn.ConvTranspose2d(nz, ngf * mult, 4, 1, 0, bias=False)),  # 1 -> 4
            _init(nn.BatchNorm2d(ngf * mult)),
            nn.ReLU(True),
        ]
        for _ in range(n):  # 4 -> 8 -> ... -> img_size/2
            layers += [
                _init(nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 4, 2, 1, bias=False)),
                _init(nn.BatchNorm2d(ngf * mult // 2)),
                nn.ReLU(True),
            ]
            mult //= 2
        layers += [_init(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.nz = nz

    def forward(self, z):
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.net(z)


class Discriminator(nn.Module):
    """Outputs raw logits (use BCEWithLogits or hinge loss)."""

    def __init__(self, ndf=64, nc=3, img_size=64, use_sn=False):
        super().__init__()
        n = _n_blocks(img_size)
        wrap = spectral_norm if use_sn else (lambda m: m)
        layers = [wrap(_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))), nn.LeakyReLU(0.2, True)]
        mult = 1
        for _ in range(n):
            layers.append(wrap(_init(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2, 1, bias=False))))
            if not use_sn:  # SN replaces BN in D
                layers.append(_init(nn.BatchNorm2d(ndf * mult * 2)))
            layers.append(nn.LeakyReLU(0.2, True))
            mult *= 2
        layers.append(wrap(_init(nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=False))))  # 4x4 -> 1
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def build_models(args):
    G = Generator(args.nz, args.ngf, 3, args.img_size)
    D = Discriminator(args.ndf, 3, args.img_size, use_sn=args.sn)
    return G, D
