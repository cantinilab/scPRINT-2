#!/usr/bin/env python3
"""Fail closed unless scPRINT uses PyTorch native SDPA attention."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - dependency-free unit imports

    class Callback:  # type: ignore[no-redef]
        pass


_NATIVE_CLASSES = {"SelfAttention", "CrossAttention"}
_LEGACY_CLASSES = {"FlashSelfAttention", "FlashCrossAttention"}


def assert_native_sdpa_attention(model: Any) -> dict[str, Any]:
    """Prove that every active attention kernel is PyTorch native SDPA."""
    attention = getattr(model, "attention", None)
    native: list[str] = []
    legacy: list[str] = []
    unexpected: list[dict[str, str]] = []
    roots = [
        (name, getattr(model, name))
        for name in ("transformer", "cell_transformer")
        if getattr(model, name, None) is not None
    ]
    for root_name, root in roots:
        for name, module in root.named_modules():
            if not name.endswith(("inner_attn", "inner_cross_attn")):
                continue
            path = f"{root_name}.{name}"
            class_name = type(module).__name__
            if class_name in _NATIVE_CLASSES:
                native.append(path)
            elif class_name in _LEGACY_CLASSES:
                legacy.append(path)
            else:
                unexpected.append({"path": path, "class": class_name})
    if attention != "normal" or legacy or unexpected or not native:
        raise RuntimeError(
            "stable attention gate rejected model: legacy Triton FlashAttention "
            f"or unexpected attention active; attention={attention!r} "
            f"legacy={legacy!r} unexpected={unexpected!r} native={native!r}"
        )
    return {
        "attention": attention,
        "native_sdpa_modules": sorted(native),
        "legacy_triton_modules": sorted(legacy),
        "unexpected_modules": unexpected,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class StableAttentionGuard(Callback):
    """Persist the native-SDPA proof before training can begin."""

    def __init__(self, receipt_path: str):
        super().__init__()
        self.receipt_path = Path(receipt_path)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        payload = {
            "status": "accepted_native_sdpa",
            "rank": int(getattr(trainer, "global_rank", 0)),
            "world_size": int(getattr(trainer, "world_size", 1)),
            "updated_at_epoch": time.time(),
            **assert_native_sdpa_attention(pl_module),
        }
        rank_path = self.receipt_path.with_name(
            f"{self.receipt_path.stem}.rank_{payload['rank']}{self.receipt_path.suffix}"
        )
        _atomic_json(rank_path, payload)
