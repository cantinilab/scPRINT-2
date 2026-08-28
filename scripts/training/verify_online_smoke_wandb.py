#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

GPU_KEYS = [
    f"fallback/gpu{gpu}/{metric}"
    for gpu in (0, 1)
    for metric in ("utilization", "memory_used_mb", "memory_total_mb", "power_watts")
]


def finite_numbers(row: dict[str, Any], keys: list[str]) -> bool:
    return all(
        key not in row
        or not isinstance(row[key], (int, float))
        or math.isfinite(float(row[key]))
        for key in keys
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_online_rows(run: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    telemetry_keys = [
        "_step",
        "fallback/trainer_global_step",
        "fallback/gpu_count",
        *GPU_KEYS,
    ]
    telemetry = [
        row
        for row in run.scan_history(keys=telemetry_keys)
        if isinstance(row.get("fallback/trainer_global_step"), (int, float))
    ]
    training = list(
        run.scan_history(keys=["trainer/global_step", "train_loss", "val_loss"])
    )
    return telemetry, training


def validate(
    run: Any,
    local_gpu_receipt: Path,
    required_step: int,
) -> dict[str, Any]:
    telemetry, training = read_online_rows(run)
    if len(telemetry) < 4:
        raise RuntimeError(f"online telemetry rows missing: {len(telemetry)}")
    if not finite_numbers(telemetry[0], GPU_KEYS):
        raise RuntimeError("non-finite online GPU metric")
    telemetry_steps: list[int] = []
    for row in telemetry:
        missing = [
            key for key in GPU_KEYS if not isinstance(row.get(key), (int, float))
        ]
        if missing:
            raise RuntimeError(f"online fallback GPU families incomplete: {missing}")
        if int(row.get("fallback/gpu_count", -1)) != 2:
            raise RuntimeError(f"online GPU count mismatch: {row}")
        trainer_step = int(row["fallback/trainer_global_step"])
        wandb_step = int(row["_step"])
        if trainer_step != wandb_step:
            raise RuntimeError(
                f"online trainer-step parity failed: trainer={trainer_step} wandb={wandb_step}"
            )
        telemetry_steps.append(trainer_step)
    local_rows = [
        json.loads(line) for line in local_gpu_receipt.read_text().splitlines()
    ]
    local_steps = [int(row["step"]) for row in local_rows]
    if sorted(local_steps) != sorted(telemetry_steps):
        raise RuntimeError(
            f"local/online telemetry parity failed: local={local_steps} online={telemetry_steps}"
        )
    trainer_steps = [
        int(row["trainer/global_step"])
        for row in training
        if isinstance(row.get("trainer/global_step"), (int, float))
    ]
    loss_keys = ["train_loss", "val_loss"]
    if not all(finite_numbers(row, loss_keys) for row in training):
        raise RuntimeError("online training history contains a non-finite loss")
    if not trainer_steps or max(trainer_steps) < required_step - 1:
        raise RuntimeError(f"online trainer steps below gate: {trainer_steps[-10:]}")
    if max(telemetry_steps) < required_step:
        raise RuntimeError(f"online telemetry did not reach step {required_step}")
    return {
        "status": "accepted",
        "run_id": run.id,
        "url": run.url,
        "state": run.state,
        "required_step": required_step,
        "trainer_steps_max": max(trainer_steps),
        "telemetry_steps": telemetry_steps,
        "trainer_step_parity": True,
        "local_online_telemetry_parity": True,
        "fallback_gpu_families": {
            "fallback/gpu0/": True,
            "fallback/gpu1/": True,
        },
        "history_rows": len(training),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("local_gpu_receipt", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--entity", default="ml4ig")
    parser.add_argument("--project", default="scprint_v2")
    parser.add_argument("--required-step", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--retry-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    import wandb

    api = wandb.Api(timeout=120)
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
            result = validate(run, args.local_gpu_receipt, args.required_step)
            atomic_json(args.receipt, result)
            print("ONLINE_SMOKE_WANDB_PASS " + json.dumps(result, sort_keys=True))
            return
        except (AssertionError, RuntimeError, ValueError) as error:
            last_error = error
            if attempt == args.attempts:
                break
            time.sleep(args.retry_seconds)
    raise RuntimeError(f"online W&B readback gate failed: {last_error}")


if __name__ == "__main__":
    main()
