#!/usr/bin/env python3
"""Deterministic, JSON-safe numerical evidence for generation tensors."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any


def _json_number(value: Any) -> int | float | str:
    """Return finite numbers unchanged and non-finite values as stable strings."""
    number = value.item() if hasattr(value, "item") else value
    if isinstance(number, bool):
        return int(number)
    number = float(number)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def summarize_tensor(value: Any) -> dict[str, Any]:
    """Summarize a tensor without mutating it or accepting hidden non-finites."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    detached = value.detach().cpu().contiguous()
    finite_mask = torch.isfinite(detached)
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(finite_mask.all()),
        "numel": detached.numel(),
        "sha256": hashlib.sha256(
            detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    if detached.numel() == 0:
        return summary
    summary.update(
        {
            "min": _json_number(detached.min()),
            "max": _json_number(detached.max()),
            "norm": _json_number(torch.linalg.vector_norm(detached.float())),
        }
    )
    if not summary["finite"]:
        index = (~finite_mask).nonzero(as_tuple=False)[0].tolist()
        summary["first_nonfinite_index"] = index
        summary["first_nonfinite_value"] = _json_number(detached[tuple(index)])
    return summary


def batch_fingerprint(
    batch: dict[str, Any], keys: tuple[str, ...] = ("x", "genes", "dataset")
) -> dict[str, str]:
    """Hash the exact source tensors used to identify a rank-local batch."""
    import torch

    result: dict[str, str] = {}
    for key in keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            raw = value.detach().cpu().contiguous().numpy().tobytes()
            result[key] = hashlib.sha256(raw).hexdigest()
    return result


class GenerationNumericsRecorder:
    """Record ordered generation boundaries and retain the earliest failure."""

    def __init__(self) -> None:
        self.timeline: list[dict[str, Any]] = []
        self.failure: dict[str, Any] | None = None

    def record(self, stage: str, value: Any) -> None:
        if self.failure is not None:
            return
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
        finite = bool(torch.isfinite(value).all())
        entry: dict[str, Any] = {
            "stage": stage,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": finite,
        }
        if value.numel():
            entry.update(
                {
                    "min": _json_number(value.detach().min()),
                    "max": _json_number(value.detach().max()),
                    "norm": _json_number(
                        torch.linalg.vector_norm(value.detach().float())
                    ),
                }
            )
        self.timeline.append(entry)
        if not finite:
            self.failure = {"stage": stage, "summary": summarize_tensor(value)}


def first_nonfinite(
    stages: Iterable[tuple[str, Any]],
) -> dict[str, Any] | None:
    """Return the first non-finite tensor in caller-supplied execution order."""
    for stage, value in stages:
        summary = summarize_tensor(value)
        if not summary["finite"]:
            return {"stage": stage, "summary": summary}
    return None
