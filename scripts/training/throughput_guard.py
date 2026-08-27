from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:

    class Callback:
        pass


class ThroughputGuard(Callback):
    def __init__(
        self,
        receipt_path: str,
        expected_steps: int,
        max_seconds_per_step: float,
        batch_size: int,
        world_size: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        self.path = Path(receipt_path)
        self.expected_steps = expected_steps
        self.max_seconds_per_step = max_seconds_per_step
        self.batch_size = batch_size
        self.world_size = world_size
        self.clock = clock
        self.started = 0.0
        self.steps = 0
        self.receipt: dict[str, Any] = {}

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.started = self.clock()

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self.steps += 1
        if self.steps != self.expected_steps:
            return
        elapsed = self.clock() - self.started
        sec = elapsed / self.steps
        world = self.world_size or int(getattr(trainer, "world_size", 1))
        rank = int(getattr(trainer, "global_rank", 0))
        self.receipt = {
            "status": "accepted" if sec <= self.max_seconds_per_step else "rejected",
            "accepted": sec <= self.max_seconds_per_step,
            "rank": rank,
            "world_size": world,
            "steps": self.steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": sec,
            "samples_per_second_global": self.batch_size * world / sec,
            "max_seconds_per_step": self.max_seconds_per_step,
        }
        path = self.path.with_name(f"{self.path.stem}.rank_{rank}{self.path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
        if not self.receipt["accepted"]:
            raise RuntimeError(f"throughput regression: {self.receipt}")
