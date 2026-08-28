#!/usr/bin/env python3
"""Append immutable hourly online W&B readbacks for one exact lineage."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Iterable


def validate_history_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reject non-finite numeric history and report observed progress."""
    finite_rows = 0
    max_global_step = -1
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"non-finite W&B metric row={row_index} key={key} value={value}"
                )
        step = row.get("trainer/global_step", row.get("global_step", row.get("_step")))
        if isinstance(step, (int, float)) and math.isfinite(float(step)):
            max_global_step = max(max_global_step, int(step))
        finite_rows += 1
    return {"finite_rows": finite_rows, "max_global_step": max_global_step}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def readback(entity: str, project: str, run_id: str) -> dict[str, Any]:
    import wandb

    run = wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
    rows = list(run.scan_history(page_size=1000))
    evidence = validate_history_rows(rows)
    return {
        "status": "online_readback_finite",
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_name": run.name,
        "run_state": run.state,
        "run_url": run.url,
        "history_rows": len(rows),
        "readback_at_epoch": time.time(),
        **evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("audit_jsonl", type=Path)
    parser.add_argument("status_json", type=Path)
    parser.add_argument("--entity", default="jkobject")
    parser.add_argument("--project", default="scprint_v2")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        try:
            payload = readback(args.entity, args.project, args.run_id)
            append_jsonl(args.audit_jsonl, payload)
            atomic_json(args.status_json, payload)
            delay = args.interval_seconds
        except Exception as exc:
            payload = {
                "status": "online_readback_retry",
                "entity": args.entity,
                "project": args.project,
                "run_id": args.run_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[-2000:],
                "readback_at_epoch": time.time(),
            }
            atomic_json(args.status_json, payload)
            delay = args.retry_seconds
        deadline = time.monotonic() + delay
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
