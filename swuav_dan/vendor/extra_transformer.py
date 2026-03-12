from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ultralytics.nn.modules.conv import Conv

__all__ = ["AttentionTSSA", "PSABlock", "C2PSA", "C2ACT"]


class AttentionTSSA(nn.Module):
    # Adapted from the ToST-style attention used in the reference YOLO11 fork.
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        w_normed = F.normalize(w, dim=-2)
        w_sq = w_normed**2
        pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)
        dots = torch.matmul((pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w**2)
        attn = self.attn_drop(1.0 / (1.0 + dots))
        out = -(w * pi.unsqueeze(-1)) * attn
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.add = shortcut
        self.attn = AttentionTSSA(c, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn_in = x.flatten(2).permute(0, 2, 1)
        attn_out = self.attn(attn_in).permute(0, 2, 1).view(b, c, h, w).contiguous()
        x = x + attn_out if self.add else attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out if self.add else ffn_out
        return x


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2ACT(C2PSA):
    pass
