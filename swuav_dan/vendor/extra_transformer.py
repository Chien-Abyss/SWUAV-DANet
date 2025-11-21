from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["AttentionTSSA", "PSABlock", "C2PSA", "C2ACT"]


class AttentionTSSA(nn.Module):
    def __init__(self, c: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(c, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.add = shortcut
        self.attn = AttentionTSSA(c, num_heads=num_heads)
        self.ffn = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn_in = x.flatten(2).transpose(1, 2)
        attn_out = self.attn(attn_in).transpose(1, 2).view(b, c, h, w)
        x = x + attn_out if self.add else attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out if self.add else ffn_out
        return x


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1, 1)
        self.c = c_
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2ACT(C2PSA):
    pass
