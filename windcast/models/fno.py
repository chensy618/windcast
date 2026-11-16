"""Fourier neural operator (Li et al. 2021) and its U-FNO variant (Wen et al. 2022: the last
`unet_layers` Fourier layers get a parallel U-Net branch for local structure).

The domain is not periodic, so the input is zero-padded by `pad` cells on the high side
before the FFT and cropped afterwards. Spectral weights and FFTs run in float32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from windcast.models.unet import UNet


class SpectralConv2d(nn.Module):
    """Learned complex weights on the lowest `modes` frequencies of each axis."""

    def __init__(self, cin: int, cout: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w_pos = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        m = self.modes
        with torch.autocast(x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float())
            out = torch.zeros(x.shape[0], self.w_pos.shape[1], H, W // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
            out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w_pos)
            out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w_neg)
            return torch.fft.irfft2(out, s=(H, W))


class FNO(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, modes: int = 32,
                 layers: int = 4, unet_layers: int = 0, pad: int = 24):
        super().__init__()
        self.pad = pad
        self.lift = nn.Sequential(nn.Conv2d(n_in, 2 * width, 1), nn.GELU(),
                                  nn.Conv2d(2 * width, width, 1))
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.first_unet = layers - unet_layers
        self.unets = nn.ModuleList(UNet(width, width, width // 2, depth=3, zero_out=False)
                                   for _ in range(unet_layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 4 * width, 1), nn.GELU(),
                                  nn.Conv2d(4 * width, n_out, 1))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        h = F.pad(self.lift(x), (0, self.pad, 0, self.pad))
        n = len(self.spectral)
        for i in range(n):
            y = self.spectral[i](h) + self.pointwise[i](h)
            if i >= self.first_unet:
                y = y + self.unets[i - self.first_unet](h)
            h = F.gelu(y) if i < n - 1 else y
        return self.proj(h[..., :H, :W])
