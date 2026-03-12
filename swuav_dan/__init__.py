from __future__ import annotations

import os
from pathlib import Path

from .registry import register

# Keep Ultralytics runtime files inside the repository by default to avoid
# permission issues on locked-down user profiles.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".yolo_cfg"))

register()

__all__ = ["register"]
