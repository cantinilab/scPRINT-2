from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_guard_class():
    path = Path(__file__).resolve().parents[1] / "scripts" / "training"
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(
            "generation_intermediate_guard", path / "generation_intermediate_guard.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.GenerationIntermediateGuard
    finally:
        sys.path.remove(str(path))


def test_bounded_instrumentation_window_does_not_stop_long_training(tmp_path):
    guard_class = load_guard_class()
    guard = guard_class(
        receipt_path=str(tmp_path / "generation.json"),
        batch_trace_path=str(tmp_path / "batch.jsonl"),
        instrument_from_step=0,
        instrument_until_step=300,
        stop_after_steps=300,
        stop_training=False,
    )
    assert guard._instrumentation_active(0)
    assert guard._instrumentation_active(299)
    assert not guard._instrumentation_active(300)

    trainer = SimpleNamespace(global_step=300, should_stop=False)
    guard.on_train_batch_end(trainer, None, None, None, 300)
    assert trainer.should_stop is False
