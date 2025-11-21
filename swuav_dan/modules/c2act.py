from __future__ import annotations

import torch
import torch.nn as nn

from swuav_dan.vendor.extra_transformer import AttentionTSSA, C2PSA, PSABlock

__all__ = ["DynamicTanh", "TSSAlock_DYT", "C2ACT"]


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape: int, channels_last: bool, alpha_init_value: float = 0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class TSSAlock_DYT(PSABlock):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = x.size()
        attn_in = self.dyt1(x).flatten(2).permute(0, 2, 1)
        attn_out = self.attn(attn_in).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        x = x + attn_out if self.add else attn_out

        ffn_out = self.ffn(self.dyt2(x))
        x = x + ffn_out if self.add else ffn_out
        return x


class C2ACT(C2PSA):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
