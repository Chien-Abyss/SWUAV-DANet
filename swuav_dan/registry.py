from __future__ import annotations

from typing import Any

__all__ = ["register"]

_REGISTERED = False


def _extend_unique(module: Any, attr: str, value: Any) -> None:
    seq = getattr(module, attr, None)
    if not seq or value in seq:
        return
    setattr(module, attr, tuple(seq) + (value,))


def _patch_parse_model(tasks_module: Any, detect_cls: Any, c2act_cls: Any) -> None:
    if getattr(tasks_module, "_swuav_danet_patched", False):
        return

    import ast
    import contextlib
    import torch
    import torch.nn as nn

    LOGGER = tasks_module.LOGGER
    colorstr = tasks_module.colorstr
    make_divisible = tasks_module.make_divisible
    task_globals = tasks_module.__dict__

    core_blocks = {
        tasks_module.Classify,
        tasks_module.Conv,
        tasks_module.ConvTranspose,
        tasks_module.GhostConv,
        tasks_module.Bottleneck,
        tasks_module.GhostBottleneck,
        tasks_module.SPP,
        tasks_module.SPPF,
        tasks_module.DWConv,
        tasks_module.Focus,
        tasks_module.BottleneckCSP,
        tasks_module.C1,
        tasks_module.C2,
        tasks_module.C2f,
        tasks_module.RepNCSPELAN4,
        tasks_module.ADown,
        tasks_module.SPPELAN,
        tasks_module.C2fAttn,
        tasks_module.C3,
        tasks_module.C3TR,
        tasks_module.C3Ghost,
        torch.nn.ConvTranspose2d,
        tasks_module.DWConvTranspose2d,
        tasks_module.C3x,
        tasks_module.RepC3,
        c2act_cls,
    }
    detect_blocks = {
        cls
        for cls in (
            task_globals.get("Detect"),
            task_globals.get("WorldDetect"),
            task_globals.get("Segment"),
            task_globals.get("Pose"),
            task_globals.get("OBB"),
            task_globals.get("ImagePoolingAttn"),
            detect_cls,
        )
        if cls is not None
    }

    def parse_model(d: dict, ch: int, verbose: bool = True):
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            tasks_module.Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            if isinstance(m, str):
                m = getattr(torch.nn, m[3:]) if "nn." in m else task_globals[m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            if m in {tasks_module.C3, tasks_module.C3Ghost, tasks_module.C3TR, tasks_module.C3x, tasks_module.RepC3}:
                if len(args) == 3 and isinstance(args[2], (float, int)) and not isinstance(args[2], bool):
                    args = [args[0], args[1], 1, args[2]]

            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in core_blocks:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is tasks_module.C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )

                args = [c1, c2, *args[1:]]
                if m in {
                    tasks_module.BottleneckCSP,
                    tasks_module.C1,
                    tasks_module.C2,
                    tasks_module.C2f,
                    tasks_module.C2fAttn,
                    tasks_module.C3,
                    tasks_module.C3TR,
                    tasks_module.C3Ghost,
                    tasks_module.C3x,
                    tasks_module.RepC3,
                    c2act_cls,
                }:
                    args.insert(2, n)
                    n = 1
            elif m is tasks_module.AIFI:
                args = [ch[f], *args]
            elif m in {tasks_module.HGStem, tasks_module.HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is tasks_module.HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is tasks_module.ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is tasks_module.Concat:
                c2 = sum(ch[x] for x in f)
            elif m in detect_blocks:
                if len(args) > 1 and isinstance(args[1], (int, float)) and args[1] != nc:
                    args[1] = make_divisible(min(args[1], max_channels) * width, 8)
                args.append([ch[x] for x in f])
                if m is task_globals.get("Segment"):
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            elif m is tasks_module.RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m is tasks_module.CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is tasks_module.CBFuse:
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    tasks_module.parse_model = parse_model
    tasks_module._swuav_danet_patched = True


def register(force: bool = False) -> None:
    global _REGISTERED
    if _REGISTERED and not force:
        return

    from pathlib import Path
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sibling_ultralytics = repo_root.parent / "ultralytics"
    if sibling_ultralytics.exists() and str(sibling_ultralytics) not in sys.path:
        sys.path.append(str(sibling_ultralytics))

    try:
        import ultralytics.nn.modules as ulty_modules
        import ultralytics.nn.tasks as tasks
    except ModuleNotFoundError:
        import site
        import sysconfig

        for p in site.getsitepackages():
            if p not in sys.path:
                sys.path.append(p)
        for key in ("purelib", "platlib"):
            p = sysconfig.get_paths().get(key)
            if p and p not in sys.path:
                sys.path.append(p)
        for base in (Path(sys.prefix), Path(sys.base_prefix)):
            candidate = base / "Lib" / "site-packages"
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        try:
            import ultralytics.nn.modules as ulty_modules
            import ultralytics.nn.tasks as tasks
        except ModuleNotFoundError as exc:
            raise ImportError(
                "SWUAV-DANet requires Ultralytics to be installed and importable before registration."
            ) from exc

    from ultralytics.nn.modules.block import C3

    setattr(ulty_modules, "C3k2", C3)
    setattr(tasks, "C3k2", C3)

    import sys
    from .vendor import extra_head as ulty_head, extra_transformer as ulty_transformer

    sys.modules["ultralytics.nn.extra_modules.head"] = ulty_head
    sys.modules["ultralytics.nn.extra_modules.transformer"] = ulty_transformer

    from .heads import Detect_DAAH
    from .modules import C2ACT

    for target in (tasks, ulty_modules, ulty_transformer):
        setattr(target, "C2ACT", C2ACT)
    for target in (tasks, ulty_head):
        setattr(target, "Detect_DAAH", Detect_DAAH)

    _extend_unique(tasks, "C2PSA_CLASS", C2ACT)
    _extend_unique(tasks, "DETECT_CLASS", Detect_DAAH)
    _patch_parse_model(tasks, Detect_DAAH, C2ACT)

    _REGISTERED = True
