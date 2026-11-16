"""U-Net: the convolutional reference in the comparison (a fixed-resolution network, not
an operator)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class UNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 32, depth: int = 4,
                 zero_out: bool = True):
        super().__init__()
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.depth = depth
        self.enc = nn.ModuleList([conv_block(n_in, ch[0])] +
                                 [conv_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2)
                                for i in reversed(range(depth)))
        self.dec = nn.ModuleList(conv_block(2 * ch[i], ch[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(ch[0], n_out, 1)
        if zero_out:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        k = 2 ** self.depth
        x = F.pad(x, (0, (-W) % k, 0, (-H) % k), mode="replicate")
        skips = [self.enc[0](x)]
        for enc in self.enc[1:]:
            skips.append(enc(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([skips.pop(), up(h)], 1))
        return self.head(h)[..., :H, :W]
