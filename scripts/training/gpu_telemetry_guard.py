from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:

    class Callback:
        pass


def parse_nvidia_smi(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        index, utilization, used, total, power = [
            item.strip() for item in line.split(",")
        ]
        gpu = int(index)
        count += 1
        metrics[f"fallback/gpu{gpu}/utilization"] = float(utilization)
        metrics[f"fallback/gpu{gpu}/memory_used_mb"] = float(used)
        metrics[f"fallback/gpu{gpu}/memory_total_mb"] = float(total)
        metrics[f"fallback/gpu{gpu}/power_watts"] = float(power)
    metrics["fallback/gpu_count"] = count
    if count != 2:
        raise RuntimeError(f"expected two directly visible GPUs, got {metrics}")
    return metrics


class GPUTelemetryGuard(Callback):
    def __init__(self, receipt_path: str, every_n_steps: int = 100):
        super().__init__()
        self.path = Path(receipt_path)
        self.every = every_n_steps

    def capture(self, trainer: Any) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        process = subprocess.run(
            command, text=True, capture_output=True, check=True, timeout=20
        )
        metrics = parse_nvidia_smi(process.stdout)
        step = int(getattr(trainer, "global_step", 0))
        metrics["fallback/trainer_global_step"] = step
        logger = getattr(trainer, "logger", None)
        if logger in {None, False}:
            raise RuntimeError("online GPU telemetry requires an active logger")
        logger.log_metrics(metrics, step=step)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"epoch": time.time(), "step": step, "metrics": metrics}
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self.capture(trainer)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if int(getattr(trainer, "global_step", 0)) % self.every == 0:
            self.capture(trainer)
