from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_loss_module():
    path = Path(__file__).parents[1] / "scprint2" / "model" / "loss.py"
    spec = importlib.util.spec_from_file_location("scprint2_loss_bf16_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_bf16_logits_with_parent_targets_stay_finite():
    loss = _load_loss_module()
    pred = torch.zeros((2, 2), dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([2, 3])
    hierarchy = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    value = loss.hierarchical_classification(
        pred=pred, cl=labels, labels_hierarchy=hierarchy
    )

    assert value.dtype == torch.float32
    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_zero_logits_are_not_treated_as_masked_logits():
    loss = _load_loss_module()
    labels = torch.tensor([2])
    hierarchy = torch.tensor([[1, 1]], dtype=torch.bool)

    zero = loss.hierarchical_classification(
        pred=torch.zeros((1, 2), dtype=torch.bfloat16),
        cl=labels,
        labels_hierarchy=hierarchy,
    )
    epsilon = loss.hierarchical_classification(
        pred=torch.full((1, 2), 1e-3, dtype=torch.float32),
        cl=labels,
        labels_hierarchy=hierarchy,
    )

    assert torch.isfinite(zero)
    assert torch.isfinite(epsilon)
    assert abs(float(zero - epsilon)) < 1e-2


def test_parent_without_children_fails_closed():
    loss = _load_loss_module()
    with pytest.raises(ValueError, match="parent without children"):
        loss.hierarchical_classification(
            pred=torch.zeros((1, 2), dtype=torch.bfloat16),
            cl=torch.tensor([2]),
            labels_hierarchy=torch.tensor([[0, 0]], dtype=torch.bool),
        )
