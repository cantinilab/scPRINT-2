#!/usr/bin/env python3
"""Fail-fast finite checks for scPRINT training and validation."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

try:  # Tests exercise the dependency-free core on the Mac.
    import torch
except ImportError:  # pragma: no cover - production environment has torch
    torch = None

try:  # Lightning is only required when the callback is instantiated remotely.
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - dependency-free unit tests
    class Callback:  # type: ignore[no-redef]
        pass


class NonFiniteError(RuntimeError):
    """Raised at the first named non-finite value."""


def decode_wandb_number(value: Any) -> Any:
    """Decode W&B's JSON representations without accepting NaN strings."""
    if isinstance(value, str) and value.strip().lower() in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return float(value)
    return value


def clamp_positive_denominator(value: float, eps: float = 1e-8) -> float:
    """Define normalization for biologically valid all-zero expression rows."""
    return max(float(value), eps)


def _tensor_failure(value: Any) -> str | None:
    if torch is None or not isinstance(value, torch.Tensor):
        return None
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return None
    first = (~finite).nonzero(as_tuple=False)[0].tolist()
    bad = value[tuple(first)].detach().cpu().item()
    return f"{bad!r} at index={first} dtype={value.dtype} shape={list(value.shape)}"


def assert_finite_tree(value: Any, *, stage: str) -> None:
    """Walk nested outputs and name the first non-finite component."""
    tensor_failure = _tensor_failure(value)
    if tensor_failure is not None:
        raise NonFiniteError(f"{stage}={tensor_failure}")
    if torch is not None and isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, stage=f"{stage}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, stage=f"{stage}[{index}]")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise NonFiniteError(f"{stage}={value}")


def assert_nonempty_tensors(
    tensors: dict[str, Any], *, stage: str, rank: int, call: str
) -> None:
    """Reject undefined empty reductions and report every operand shape."""
    shapes = " ".join(
        f"{name}_shape={list(tensor.shape)}" for name, tensor in tensors.items()
    )
    for name, tensor in tensors.items():
        if tensor.numel() == 0:
            raise NonFiniteError(
                f"{stage}.empty rank={rank} call={call} component={name} {shapes}"
            )


def batch_has_empty_reduction_domain(batch: Any) -> bool:
    """Return True only when a rank-local batch has an undefined empty axis."""
    if torch is None or not isinstance(batch, dict):
        return False
    expression = batch.get("x")
    return bool(
        isinstance(expression, torch.Tensor)
        and (expression.ndim < 2 or expression.shape[0] == 0 or expression.shape[1] == 0)
    )


def zero_safe_mse(input: Any, target: Any) -> Any:
    """scPRINT MSE with defined normalization for zero-count rows."""
    global _MSE_CALL_INDEX
    if torch is None:
        raise RuntimeError("zero_safe_mse requires torch")
    _MSE_CALL_INDEX += 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    call_names = {1: "mask", 2: "denoise", 3: "generate"}
    call = call_names.get(_MSE_CALL_INDEX, f"call_{_MSE_CALL_INDEX}")
    assert_nonempty_tensors(
        {"input": input, "target": target}, stage="mse", rank=rank, call=call
    )
    assert_finite_tree(input, stage=f"mse.rank_{rank}.{call}.raw_prediction")
    assert_finite_tree(target, stage=f"mse.rank_{rank}.{call}.raw_target")
    if bool((input < 0).any()):
        raise NonFiniteError(f"mse.raw_prediction_negative_min={input.min().item()}")
    if bool((target < 0).any()):
        raise NonFiniteError(f"mse.raw_target_negative_min={target.min().item()}")
    input = torch.log2(input + 1)
    target = torch.log2(target + 1)
    assert_finite_tree(input, stage="mse.logged_prediction")
    assert_finite_tree(target, stage="mse.logged_target")
    input_denominator = torch.sum(input, dim=1, keepdim=True)
    target_denominator = torch.sum(target, dim=1, keepdim=True)
    assert_finite_tree(input_denominator, stage="mse.prediction_denominator")
    assert_finite_tree(target_denominator, stage="mse.target_denominator")
    input = input / torch.clamp(input_denominator, min=1e-8) * 10000
    target = target / torch.clamp(target_denominator, min=1e-8) * 10000
    assert_finite_tree(input, stage="mse.normalized_prediction")
    assert_finite_tree(target, stage="mse.normalized_target")
    result = torch.nn.functional.mse_loss(input, target, reduction="mean")
    assert_finite_tree(result, stage="mse.zero_safe_result")
    return result


zero_safe_mse._revision2_zero_safe = True  # type: ignore[attr-defined]
_MSE_CALL_INDEX = 0
_ZINB_CALL_INDEX = 0


def import_scprint_loss() -> Any:
    """Import the loss module from current scPRINT-2 or the legacy package."""
    try:
        from scprint2.model import loss as scprint_loss
    except ImportError:
        from scprint.model import loss as scprint_loss
    return scprint_loss



def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class FiniteTrainingGuard(Callback):
    """Abort before optimizer/output corruption can waste an allocation."""

    def __init__(self, receipt_path: str, required_train_steps: int = 200):
        super().__init__()
        self.receipt_path = Path(receipt_path)
        self.required_train_steps = required_train_steps
        self.train_batches = 0
        self.gradient_checks = 0
        self.parameter_checks = 0
        self.validation_batches = 0
        self.validation_embeddings_checked = False
        self.coordinated_skipped_batches = 0
        self.started_at = time.time()
        self.last_failure_context: dict[str, Any] = {}

    @staticmethod
    def _batch_summary(batch: Any) -> dict[str, Any]:
        if torch is None or not isinstance(batch, dict):
            return {"type": type(batch).__name__}
        import hashlib

        summary: dict[str, Any] = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                summary[key] = {"type": type(value).__name__}
                continue
            detached = value.detach().cpu().contiguous()
            item: dict[str, Any] = {
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "finite": bool(torch.isfinite(detached).all()),
                "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
            }
            if detached.numel():
                item.update(
                    {
                        "min": detached.min().item(),
                        "max": detached.max().item(),
                        "negative_count": int((detached < 0).sum().item()),
                        "zero_count": int((detached == 0).sum().item()),
                    }
                )
                if detached.ndim >= 2:
                    item["all_zero_rows"] = int(
                        (detached.reshape(detached.shape[0], -1).abs().sum(1) == 0)
                        .sum()
                        .item()
                    )
            if key in {"dataset", "depth", "class", "is_meta"} and detached.numel() <= 512:
                item["values"] = detached.tolist()
            summary[key] = item
        return summary

    def _write(self, trainer: Any, status: str, **extra: Any) -> None:
        rank = (
            torch.distributed.get_rank()
            if torch is not None and torch.distributed.is_initialized()
            else int(getattr(trainer, "global_rank", 0))
        )
        receipt_path = self.receipt_path
        if not getattr(trainer, "is_global_zero", True):
            receipt_path = receipt_path.with_name(
                f"{receipt_path.stem}.rank_{rank}{receipt_path.suffix}"
            )
        payload = {
            "status": status,
            "rank": rank,
            "global_step": int(getattr(trainer, "global_step", 0)),
            "train_batches": self.train_batches,
            "gradient_checks": self.gradient_checks,
            "parameter_checks": self.parameter_checks,
            "validation_batches": self.validation_batches,
            "validation_embeddings_finite": self.validation_embeddings_checked,
            "coordinated_skipped_batches": self.coordinated_skipped_batches,
            "required_train_steps": self.required_train_steps,
            "world_size": int(getattr(trainer, "world_size", 1)),

            "root_causes": [
                "failed revision 1 used float16 AMP; ExprDecoder computes dispersion as exp(clamp(raw_variance,max=15)), above float16 log-max 11.09, and W&B persisted 200/200 non-finite train-loss rows",
                "upstream scprint.model.loss.mse divides each log-expression row by its exact row sum, so biologically valid all-zero non-empty rows produce NaN without a denominator floor",
                "revision-2 rank 1 reached generation with an undefined empty reduction domain near step 24; both upstream ZINB and MSE return NaN on shape-compatible empty tensors",
            ],
            "corrections": [
                "bf16-mixed precision on H100 preserves the decoder exponential range",
                "zero-safe per-row MSE normalization clamps only non-empty row denominators to 1e-8",
                "if any DDP rank has an empty batch axis, every rank explicitly skips that whole optimizer update before forward/backward; downstream empty ZINB/MSE remains a hard failure with operand shapes",
            ],
            "updated_at_epoch": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            **extra,
        }
        _atomic_json(receipt_path, payload)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if torch is None:
            raise RuntimeError("FiniteTrainingGuard requires torch")
        scprint_loss = import_scprint_loss()


        if not getattr(scprint_loss.zinb, "_revision2_finite_guard", False):
            original_zinb = scprint_loss.zinb

            def guarded_zinb(*, target: Any, mu: Any, theta: Any, pi: Any, eps: float = 1e-8) -> Any:
                global _ZINB_CALL_INDEX
                _ZINB_CALL_INDEX += 1
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                call_names = {1: "mask", 2: "denoise", 3: "generation"}
                call = call_names.get(_ZINB_CALL_INDEX, f"call_{_ZINB_CALL_INDEX}")
                tensors = {"target": target, "mu": mu, "theta": theta, "pi": pi}
                assert_nonempty_tensors(
                    tensors, stage="zinb", rank=rank, call=call
                )
                for name, tensor in tensors.items():
                    assert_finite_tree(tensor, stage=f"zinb.rank_{rank}.{call}.{name}")
                result = original_zinb(target=target, mu=mu, theta=theta, pi=pi, eps=eps)
                assert_finite_tree(result, stage=f"zinb.rank_{rank}.{call}.loss")
                return result

            guarded_zinb._revision2_finite_guard = True  # type: ignore[attr-defined]
            scprint_loss.zinb = guarded_zinb

        if not getattr(scprint_loss.mse, "_revision2_zero_safe", False):
            scprint_loss.mse = zero_safe_mse

        original_training_step = pl_module.training_step

        def guarded_training_step(batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
            local_empty = batch_has_empty_reduction_domain(batch)
            expression = batch.get("x") if isinstance(batch, dict) else None
            device = (
                expression.device
                if torch is not None and isinstance(expression, torch.Tensor)
                else pl_module.device
            )
            skip = torch.tensor(int(local_empty), device=device, dtype=torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(skip, op=torch.distributed.ReduceOp.MAX)
            if int(skip.item()) != 0:
                self.coordinated_skipped_batches += 1
                self._write(
                    trainer,
                    "coordinated_empty_batch_skipped",
                    local_empty_domain=local_empty,
                    batch=self._batch_summary(batch),
                )
                # Every DDP rank returns None before forward/backward. Lightning
                # therefore skips this optimizer update entirely; all-zero but
                # non-empty biological profiles still take the normal path.
                return None
            return original_training_step(batch, batch_idx, *args, **kwargs)

        pl_module.training_step = guarded_training_step

        original_full_training = pl_module._full_training

        def guarded_full_training(*args: Any, **kwargs: Any) -> Any:
            stage = "validation" if trainer.validating else "train"
            step = int(getattr(trainer, "global_step", 0))
            batch = kwargs.get("batch") if "batch" in kwargs else (args[0] if args else None)
            try:
                global _MSE_CALL_INDEX, _ZINB_CALL_INDEX
                _MSE_CALL_INDEX = 0
                _ZINB_CALL_INDEX = 0
                total, components = original_full_training(*args, **kwargs)
                assert_finite_tree(components, stage=f"{stage}.step_{step}.components")
                assert_finite_tree(total, stage=f"{stage}.step_{step}.total_loss")
                return total, components
            except NonFiniteError as exc:
                self.last_failure_context = {
                    "failure_stage": str(exc),
                    "stage": stage,
                    "global_step": step,
                    "batch": self._batch_summary(batch),
                }
                self._write(trainer, "failed_component", **self.last_failure_context)
                raise

        pl_module._full_training = guarded_full_training

    def on_before_backward(self, trainer: Any, pl_module: Any, loss: Any) -> None:
        assert_finite_tree(loss, stage=f"train.step_{trainer.global_step}.total_loss")

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is not None:
                assert_finite_tree(
                    parameter.grad,
                    stage=f"train.step_{trainer.global_step}.gradient.{name}",
                )
        self.gradient_checks += 1

    def on_before_optimizer_step(
        self, trainer: Any, pl_module: Any, optimizer: Any
    ) -> None:
        for name, parameter in pl_module.named_parameters():
            assert_finite_tree(
                parameter.data,
                stage=f"train.step_{trainer.global_step}.parameter.{name}",
            )
        self.parameter_checks += 1

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert_finite_tree(outputs, stage=f"train.step_{trainer.global_step}.output")
        metrics = {
            key: value
            for key, value in trainer.callback_metrics.items()
            if any(token in key.lower() for token in ("loss", "expr", "cce", "ecs", "vae"))
        }
        assert_finite_tree(metrics, stage=f"train.step_{trainer.global_step}.components")
        self.train_batches += 1
        if self.train_batches == 1 or self.train_batches % 10 == 0:
            self._write(trainer, "training_finite")

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert_finite_tree(outputs, stage=f"validation.batch_{batch_idx}.loss")
        assert_finite_tree(
            getattr(pl_module, "embs", None),
            stage=f"validation.batch_{batch_idx}.embeddings",
        )
        self.validation_batches += 1
        self.validation_embeddings_checked = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        assert_finite_tree(getattr(pl_module, "embs", None), stage="validation.embeddings")
        self.validation_embeddings_checked = True
        status = (
            "accepted"
            if self.train_batches >= self.required_train_steps
            else "validation_finite_but_too_few_train_steps"
        )
        self._write(trainer, status)

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self._write(
            trainer,
            "failed",
            exception_type=type(exception).__name__,
            exception=str(exception)[:4000],
            **self.last_failure_context,
        )
