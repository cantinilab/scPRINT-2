from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from finite_training_guard import assert_finite_tree

try:
    import torch
    from lightning.pytorch.callbacks import Callback
except ImportError:
    torch = None

    class Callback:
        pass


def within_bounded_probe(step: int, limit: int) -> bool:
    return int(step) < int(limit)


class InitialLongHealthGuard(Callback):
    def __init__(self, receipt_path: str, probe_steps: int = 300):
        super().__init__()
        self.path = Path(receipt_path)
        self.probe_steps = probe_steps
        self.train_batches = 0
        self.gradient_checks = 0
        self.parameter_checks = 0
        self.validation_batches = 0
        self.started = time.time()

    def _write(self, trainer: Any, status: str):
        rank = int(getattr(trainer, "global_rank", 0))
        p = self.path.with_name(f"{self.path.stem}.rank_{rank}{self.path.suffix}")
        x = {
            "status": status,
            "rank": rank,
            "world_size": int(getattr(trainer, "world_size", 1)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "train_batches": self.train_batches,
            "gradient_checks": self.gradient_checks,
            "parameter_checks": self.parameter_checks,
            "validation_batches": self.validation_batches,
            "probe_steps": self.probe_steps,
            "bounded_probe_complete": self.train_batches >= self.probe_steps,
            "elapsed_seconds": time.time() - self.started,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        t = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        t.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
        os.replace(t, p)

    def on_after_backward(self, trainer: Any, pl_module: Any):
        if not within_bounded_probe(self.gradient_checks, self.probe_steps):
            return
        for n, p in pl_module.named_parameters():
            if p.grad is not None:
                assert_finite_tree(p.grad, stage=f"long.gradient.{n}")
        self.gradient_checks += 1

    def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any):
        if not within_bounded_probe(self.parameter_checks, self.probe_steps):
            return
        for n, p in pl_module.named_parameters():
            assert_finite_tree(p.data, stage=f"long.parameter.{n}")
        self.parameter_checks += 1

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ):
        assert_finite_tree(
            outputs, stage=f"long.step_{getattr(trainer, 'global_step', 0)}.output"
        )
        self.train_batches += 1
        if self.train_batches % 50 == 0 or self.train_batches == self.probe_steps:
            self._write(
                trainer,
                "accepted_initial_probe"
                if self.train_batches >= self.probe_steps
                else "running",
            )

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        assert_finite_tree(outputs, stage=f"long.validation.{batch_idx}")
        self.validation_batches += 1

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException):
        self._write(trainer, "failed")
