from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["C3k2"]


class C3k2(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Conv(self.c, self.c, 3, 1, g=g, act=True) for _ in range(n))
        self.use_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b = y.split((self.c, self.c), dim=1)
        outputs = [b]
        for conv in self.m:
            out = conv(outputs[-1])
            if self.use_shortcut:
                out = out + outputs[-1]
            outputs.append(out)
        return self.cv2(torch.cat((a, *outputs), dim=1))
