#!/usr/bin/env python3
"""Fail-fast, stage-ordered instrumentation for scPRINT generation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from generation_numerics import (
    GenerationNumericsRecorder,
    batch_fingerprint,
    summarize_tensor,
)

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - dependency-free import on the Mac

    class Callback:  # type: ignore[no-redef]
        pass


class GenerationNonFiniteError(RuntimeError):
    """Raised at the earliest observed non-finite generation boundary."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_named_parameters(module: Any) -> dict[str, Any]:
    """Report parameter/gradient finiteness without changing optimizer state."""
    import torch

    parameter_count = 0
    gradient_count = 0
    nonfinite_parameters: list[str] = []
    nonfinite_gradients: list[str] = []
    for name, parameter in module.named_parameters():
        parameter_count += 1
        if not bool(torch.isfinite(parameter.detach()).all()):
            nonfinite_parameters.append(name)
        if parameter.grad is not None:
            gradient_count += 1
            if not bool(torch.isfinite(parameter.grad.detach()).all()):
                nonfinite_gradients.append(name)
    return {
        "parameter_count": parameter_count,
        "gradient_count": gradient_count,
        "all_parameters_finite": not nonfinite_parameters,
        "all_gradients_finite": not nonfinite_gradients,
        "nonfinite_parameters": nonfinite_parameters,
        "nonfinite_gradients": nonfinite_gradients,
    }


class GenerationIntermediateGuard(Callback):
    """Capture the first non-finite tensor in the exact generation pipeline."""

    TARGET_BATCH = {
        "x": "0b6fd2bdb07021f5f15d9bdbf4ceac64d4d13b83fd371b8826ff0d83d6d5b26a",
        "genes": "0804870de46cc6e86a7036aca498ab18a7b789a4fa1270b219c0a80726dbc65c",
        "dataset": "0f9b6d514257679b41283aef3ecc91c5b6c0f9b667a2ab1947953621aa93ecad",
    }
    REPLAY_FAILURE_BATCH = {
        "x": "261c186ef883b4ce4f092e5ad1dec467a9bdf879c80e48193fda377be0b6aa40",
        "genes": "75b6b7fa67d7017e19531bd38553a7d5de3bf357a8af7e6c74c465c572e70e48",
        "dataset": "1ac2736e77a0e9217407af830760eb364636e5161d12b38369c69c5f99c533d5",
    }

    def __init__(
        self,
        receipt_path: str,
        batch_trace_path: str,
        instrument_from_step: int = 0,
        instrument_until_step: int | None = None,
        stop_after_steps: int = 220,
        stop_training: bool = True,
    ):
        super().__init__()
        self.receipt_path = Path(receipt_path)
        self.batch_trace_path = Path(batch_trace_path)
        self.instrument_from_step = instrument_from_step
        self.instrument_until_step = instrument_until_step
        self.stop_after_steps = stop_after_steps
        self.stop_training = stop_training
        self.current_batch: Any = None
        self.current_batch_idx = -1
        self.recorder: GenerationNumericsRecorder | None = None
        self.upstream_recorder: GenerationNumericsRecorder | None = None
        self.in_generation = False
        self.instrument_enabled = False
        self.current_phase: str | None = None
        self._forward_call = 0
        self.target_batch_captured = False
        self._hooks: list[Any] = []
        self._trainer: Any = None
        self._module: Any = None
        self._vae_calls: dict[str, int] = {}

    def _instrumentation_active(self, step: int) -> bool:
        return step >= self.instrument_from_step and (
            self.instrument_until_step is None or step < self.instrument_until_step
        )

    def _rank(self) -> int:
        import torch

        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(getattr(self._trainer, "global_rank", 0))

    def _rank_path(self, path: Path) -> Path:
        rank = self._rank()
        return path.with_name(f"{path.stem}.rank_{rank}{path.suffix}")

    def _record(self, stage: str, value: Any) -> None:
        if (
            not self.instrument_enabled
            or not self.in_generation
            or self.recorder is None
        ):
            return
        self.recorder.record(stage, value)
        if self.recorder.failure is not None:
            raise GenerationNonFiniteError(
                f"{self.recorder.failure['stage']} contains a non-finite value"
            )

    def _record_upstream(self, stage: str, value: Any) -> None:
        if not self.instrument_enabled or self.upstream_recorder is None:
            return
        self.upstream_recorder.record(stage, value)
        if self.upstream_recorder.failure is not None:
            self._persist_failure()
            raise GenerationNonFiniteError(
                f"{self.upstream_recorder.failure['stage']} contains a non-finite value"
            )

    def _record_active(self, suffix: str, value: Any) -> None:
        if self.in_generation:
            self._record(f"generation.{suffix}", value)
        elif self.current_phase is not None:
            self._record_upstream(f"{self.current_phase}.{suffix}", value)

    def _record_module_output(self, stage: str, value: Any) -> None:
        """Record tensors in nested module outputs in execution order."""
        import torch

        if isinstance(value, torch.Tensor):
            self._record_active(stage, value)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self._record_module_output(f"{stage}.{index}", item)
        elif isinstance(value, dict):
            for name, item in value.items():
                self._record_module_output(f"{stage}.{name}", item)

    def _append_batch_trace(self, payload: dict[str, Any]) -> None:
        path = self._rank_path(self.batch_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _save_batch(self, label: str) -> dict[str, Any]:
        import torch

        rank = self._rank()
        path = self.receipt_path.with_name(
            f"{self.receipt_path.stem}.{label}.rank_{rank}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            key: value.detach().cpu().contiguous()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in self.current_batch.items()
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}

    def _batch_evidence(self) -> dict[str, Any]:
        import torch

        evidence: dict[str, Any] = {
            "fingerprint": batch_fingerprint(self.current_batch),
            "exact_batch": self._save_batch("failure_batch"),
        }
        for key in ("x", "genes", "depth", "dataset"):
            value = self.current_batch.get(key)
            if isinstance(value, torch.Tensor):
                evidence[key] = summarize_tensor(value)
                if value.ndim and value.shape[0] > 26:
                    evidence[f"{key}_cell_26"] = summarize_tensor(value[26])
        dataset = self.current_batch.get("dataset")
        if isinstance(dataset, torch.Tensor):
            evidence["dataset_indices"] = dataset.detach().cpu().tolist()
        return evidence

    def _persist_failure(self) -> None:
        recorder = (
            self.recorder
            if self.recorder is not None and self.recorder.failure is not None
            else self.upstream_recorder
        )
        if recorder is None or recorder.failure is None:
            return
        payload = {
            "status": "failed_generation_intermediate",
            "rank": self._rank(),
            "global_step": int(getattr(self._trainer, "global_step", -1)),
            "batch_idx": self.current_batch_idx,
            "first_nonfinite": recorder.failure,
            "timeline": recorder.timeline,
            "upstream_timeline": (
                self.upstream_recorder.timeline
                if self.upstream_recorder is not None
                else []
            ),
            "batch": self._batch_evidence(),
            "parameters": audit_named_parameters(self._module),
            "target_batch_captured": self.target_batch_captured,
            "updated_at_epoch": time.time(),
        }
        _atomic_json(self._rank_path(self.receipt_path), payload)

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
    ) -> None:
        import torch

        self.current_batch = batch
        self.current_batch_idx = batch_idx
        self.upstream_recorder = GenerationNumericsRecorder()
        self._vae_calls = {}
        self._forward_call = 0
        self.instrument_enabled = self._instrumentation_active(
            int(getattr(trainer, "global_step", -1))
        )
        dataset = batch.get("dataset") if isinstance(batch, dict) else None
        dataset_fingerprint = (
            batch_fingerprint(batch, keys=("dataset",))
            if isinstance(batch, dict)
            else {}
        )
        trace: dict[str, Any] = {
            "rank": self._rank(),
            "global_step": int(getattr(trainer, "global_step", -1)),
            "batch_idx": batch_idx,
            "dataset_sha256": dataset_fingerprint.get("dataset"),
        }
        if isinstance(dataset, torch.Tensor):
            trace["dataset_indices"] = dataset.detach().cpu().tolist()
        targets = {
            "accepted_1279361": self.TARGET_BATCH,
            "exact_config_replay_1289292": self.REPLAY_FAILURE_BATCH,
        }
        for label, target in targets.items():
            if dataset_fingerprint.get("dataset") != target["dataset"]:
                continue
            full = batch_fingerprint(batch)
            trace["full_fingerprint"] = full
            if full == target:
                trace["failure_batch_match"] = label
                trace["exact_batch"] = self._save_batch(f"target_batch_{label}")
                self.target_batch_captured = True
            break
        self._append_batch_trace(trace)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        import torch

        self._trainer = trainer
        self._module = pl_module
        original_encoder = pl_module._encoder
        original_forward = pl_module.forward
        original_generate = pl_module._generate

        def guarded_encoder(*args: Any, **kwargs: Any) -> Any:
            result = original_encoder(*args, **kwargs)
            self._record_active("encoder.cell_embs", result[0])
            self._record_active("encoder.output", result[1])
            return result

        def guarded_forward(*args: Any, **kwargs: Any) -> Any:
            phases = ("full_forward", "mask_TF", "denoise_70")
            phase = (
                phases[self._forward_call]
                if self._forward_call < len(phases)
                else f"forward_{self._forward_call}"
            )
            self._forward_call += 1
            self.current_phase = phase
            names = ("gene_pos", "expression", "neighbors", "mask", "req_depth")
            arguments = {name: value for name, value in zip(names, args)}
            arguments.update(kwargs)
            try:
                for name in names:
                    value = arguments.get(name)
                    if isinstance(value, torch.Tensor):
                        self._record_upstream(f"{phase}.input.{name}", value)
                output = original_forward(*args, **kwargs)
                if isinstance(output, dict):
                    for name in (
                        "mean",
                        "disp",
                        "zero_logits",
                        "input_cell_embs",
                        "output_cell_embs",
                    ):
                        value = output.get(name)
                        if isinstance(value, torch.Tensor):
                            self._record_upstream(f"{phase}.output.{name}", value)
                return output
            finally:
                self.current_phase = None

        def guarded_generate(*args: Any, **kwargs: Any) -> Any:
            names = ("cell_embs", "gene_pos", "depth_mult", "req_depth")
            arguments = {name: value for name, value in zip(names, args)}
            arguments.update(kwargs)
            self.recorder = GenerationNumericsRecorder()
            self.in_generation = True
            try:
                for name in names:
                    value = arguments.get(name)
                    if isinstance(value, torch.Tensor):
                        self._record(f"generation.input.{name}", value)
                output = original_generate(*args, **kwargs)
                for name in ("mean", "disp", "zero_logits"):
                    value = output.get(name)
                    if isinstance(value, torch.Tensor):
                        self._record(f"generation.output.{name}", value)
                return output
            except GenerationNonFiniteError:
                self._persist_failure()
                raise
            finally:
                self.in_generation = False

        pl_module._encoder = guarded_encoder
        pl_module.forward = guarded_forward
        pl_module._generate = guarded_generate

        if pl_module.compressor is not None:
            for compressor_name, compressor in pl_module.compressor.items():
                if not hasattr(compressor, "reparameterize"):
                    continue
                original_reparameterize = compressor.reparameterize

                def guarded_reparameterize(
                    mu: Any,
                    log_var: Any,
                    *,
                    compressor_name: str = compressor_name,
                    original_reparameterize: Any = original_reparameterize,
                ) -> Any:
                    call = self._vae_calls.get(compressor_name, 0)
                    self._vae_calls[compressor_name] = call + 1
                    if call > 0:
                        return original_reparameterize(mu, log_var)
                    prefix = f"upstream.vae.{compressor_name}.full_forward"
                    self._record_upstream(f"{prefix}.mu", mu)
                    self._record_upstream(f"{prefix}.log_var", log_var)
                    half_log_var = 0.5 * log_var
                    self._record_upstream(f"{prefix}.half_log_var", half_log_var)
                    std = torch.exp(half_log_var)
                    self._record_upstream(f"{prefix}.std", std)
                    eps = torch.randn_like(std)
                    self._record_upstream(f"{prefix}.eps", eps)
                    latent = mu + eps * std
                    self._record_upstream(f"{prefix}.latent", latent)
                    return latent

                compressor.reparameterize = guarded_reparameterize

        def transformer_pre(_module: Any, inputs: tuple[Any, ...]) -> None:
            if inputs:
                self._record_active("transformer.input", inputs[0])

        def transformer_post(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            self._record_active("transformer.output", value)

        self._hooks.append(
            pl_module.transformer.register_forward_pre_hook(transformer_pre)
        )
        self._hooks.append(
            pl_module.transformer.register_forward_hook(transformer_post)
        )

        # The outer transformer boundary identified the active failure, so retain
        # execution-ordered boundaries around every operation that can first
        # create a non-finite inside each shared FlashTransformer block.
        selected_suffixes = {
            "norm1",
            "norm2",
            "norm3",
            "mixer.Wqkv",
            "mixer.Wq",
            "mixer.Wkv",
            "mixer.inner_attn",
            "mixer.inner_cross_attn",
            "mixer.out_proj",
            "mixer",
            "mlp.fc1",
            "mlp.activation",
            "mlp.fc2",
            "mlp",
        }
        for module_name, module in pl_module.transformer.named_modules():
            if not module_name.startswith("blocks."):
                continue
            parts = module_name.split(".")
            relative_name = ".".join(parts[2:])
            is_block = len(parts) == 2
            if not is_block and relative_name not in selected_suffixes:
                continue
            stage = f"transformer.{module_name}"

            def module_post(
                _module: Any, _inputs: Any, output: Any, *, stage: str = stage
            ) -> None:
                self._record_module_output(stage, output)

            self._hooks.append(module.register_forward_hook(module_post))
        if hasattr(pl_module.transformer, "norm"):
            self._hooks.append(
                pl_module.transformer.norm.register_forward_hook(
                    lambda _module, _inputs, output: self._record_module_output(
                        "transformer.final_norm", output
                    )
                )
            )

        def decoder_pre(_module: Any, inputs: tuple[Any, ...]) -> None:
            if inputs:
                self._record_active("expr_decoder.input", inputs[0])
            if len(inputs) > 1:
                self._record_active("req_depth_log2", inputs[1])

        self._hooks.append(
            pl_module.expr_decoder.register_forward_pre_hook(decoder_pre)
        )
        for index, layer in enumerate(pl_module.expr_decoder.fc):
            stage = f"expr_decoder.fc.{index}.{type(layer).__name__}"

            def layer_post(
                _module: Any, _inputs: Any, output: Any, *, stage: str = stage
            ) -> None:
                self._record_active(stage, output)

            self._hooks.append(layer.register_forward_hook(layer_post))

        def logits_post(_module: Any, _inputs: Any, output: Any) -> None:
            self._record_active("expr_decoder.combined_logits", output)
            pred_value, var_value, zero_logits = output.split(1, dim=-1)
            pred_value = pred_value.squeeze(-1)
            var_value = var_value.squeeze(-1)
            zero_logits = zero_logits.squeeze(-1)
            self._record_active("expr_decoder.pred_value", pred_value)
            self._record_active("expr_decoder.variance_logits", var_value)
            self._record_active("expr_decoder.zero_logits", zero_logits)
            self._record_active(
                "expr_decoder.softmax", torch.nn.functional.softmax(pred_value, dim=-1)
            )
            self._record_active(
                "expr_decoder.disp",
                torch.exp(torch.clamp(var_value, max=15)),
            )

        self._hooks.append(
            pl_module.expr_decoder.pred_var_zero.register_forward_hook(logits_post)
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.stop_training and int(
            getattr(trainer, "global_step", 0)
        ) >= self.stop_after_steps:
            trainer.should_stop = True

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
