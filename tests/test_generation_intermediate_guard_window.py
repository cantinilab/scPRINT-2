from __future__ import annotations

from types import SimpleNamespace

from generation_intermediate_guard import GenerationIntermediateGuard


def test_bounded_instrumentation_window_does_not_stop_long_training(tmp_path):
    guard = GenerationIntermediateGuard(
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
