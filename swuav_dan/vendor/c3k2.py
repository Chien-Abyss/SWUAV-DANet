from __future__ import annotations

import torch.nn as nn

from ultralytics.nn.modules.block import C2f, C3, Bottleneck

__all__ = ["C3k2"]


class C3k(C3):
    """C3 block with configurable kernel size, aligned with the YOLO11 fork behavior."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster CSP bottleneck used by YOLO11 model variants."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
