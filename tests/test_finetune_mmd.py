import ast
from pathlib import Path
from typing import Dict

import torch

from scprint2.tasks.finetune import FinetuneBatchClass

SOURCE_PATH = Path(__file__).parents[1] / "scprint2" / "tasks" / "finetune.py"


def _source():
    return SOURCE_PATH.read_text()


def _function_node(name):
    tree = ast.parse(_source())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _load_mmd_loss():
    module = ast.Module(
        body=[
            _function_node("_legacy_unbiased_energy_mmd"),
            _function_node("mmd_loss"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["mmd_loss"]


def _load_mmd_helpers():
    helper_names = {
        "_legacy_unbiased_energy_mmd",
        "mmd_loss",
        "_balanced_group_pairwise_mmd",
    }
    tree = ast.parse(_source())
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["mmd_loss"], namespace["_balanced_group_pairwise_mmd"]


def _load_legacy_mmd_helpers():
    helper_names = {
        "_legacy_detached_one_vs_rest_mmd",
        "_legacy_unbiased_energy_mmd",
    }
    tree = ast.parse(_source())
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_legacy_detached_one_vs_rest_mmd"]


def _load_trainable_state_helpers():
    helper_names = {"_trainable_state_dict", "_restore_trainable_state"}
    tree = ast.parse(_source())
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "Dict": Dict}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_trainable_state_dict"], namespace["_restore_trainable_state"]


def test_batch_correction_keeps_mmd_attached_to_autograd():
    source = ast.get_source_segment(_source(), _function_node("batch_corr_pass"))

    assert "total_loss += tot_mmd_loss *" in source
    assert "mmd_for_logging = tot_mmd_loss.detach().item()" in source
    assert "total_loss += mmd_for_logging *" in source


def test_mmd_is_active_with_point_zero_three_default_weight():
    finetuner = FinetuneBatchClass()

    assert finetuner.do_mmd_on == "cell_type_ontology_term_id"
    assert finetuner.batch_key == "batch"
    assert finetuner.loss_scalers["mmd"] == 0.03


def test_learned_batch_embedding_uses_precompression_width():
    source = _source()

    assert "embedding_dim=model.d_model" in source
    assert "compressor[self.learn_batches_on].fc_mu.weight.shape[0]" not in source


def test_batch_correction_mmd_uses_user_defined_batch_groups():
    source = ast.get_source_segment(_source(), _function_node("batch_corr_pass"))

    assert "mmd_groups = class_elem[:, self.batch_group_position]" in source
    assert "_balanced_group_pairwise_mmd(" in source
    assert "selected_emb, mmd_groups" in source
    assert "GradReverse" not in source
    assert "class_pos = self.predict_keys.index(self.do_mmd_on)" not in source
    assert "cell_labels" not in source
    assert "in_cell_type" not in source


def test_finetune_does_not_add_organism_classification_implicitly():
    source = ast.get_source_segment(_source(), _function_node("__call__"))

    assert 'self.predict_keys.append("organism_ontology_term_id")' not in source
    assert "dataset_obs_keys" in source
    assert "self.predict_keys + [self.batch_key, organism_key]" in source
    assert "self.predict_keys + [self.batch_key]" in source


def test_task3_organism_classification_keeps_organism_decoder_frozen():
    source = ast.get_source_segment(_source(), _function_node("__call__"))

    assert 'class_name == "organism_ontology_term_id"' in source
    assert "and not self.train_organism_decoder" in source


def test_xpressor_mode_unfreezes_cross_attention_parameters():
    source = ast.get_source_segment(_source(), _function_node("__call__"))

    assert "for parameter in block.cross_attn.parameters():" in source
    assert "parameter.requires_grad = True" in source
    assert "cross_attn.requires_grad = True" not in source


def test_batch_correction_passes_knn_cells_to_model_forward():
    source = ast.get_source_segment(_source(), _function_node("batch_corr_pass"))

    assert "neighbors=(" in source
    assert 'batch["knn_cells"].to(model.device)' in source
    assert "neighbors_info=(" in source
    assert 'batch["knn_cells_info"].to(model.device)' in source


def test_mmd_validation_order_is_randomized_before_dataset_creation():
    source = ast.get_source_segment(_source(), _function_node("__call__"))

    assert "if self.do_mmd_on is not None:" in source
    assert "np.random.shuffle(val_idx)" in source


def test_batch_correction_mmd_averages_unordered_species_pairs():
    _, balanced_mmd = _load_mmd_helpers()
    seen_pairs = []

    def fake_mmd(left, right):
        pair = (int(left[:, 0].mean().item()), int(right[:, 0].mean().item()))
        seen_pairs.append(pair)
        return left[:, 0].mean() * right[:, 0].mean()

    balanced_mmd.__globals__["mmd_loss"] = fake_mmd
    embeddings = torch.tensor([[10.0], [10.0], [20.0], [30.0], [30.0]])
    species = torch.tensor([1, 1, 2, 3, 3])

    value = balanced_mmd(embeddings, species)

    assert seen_pairs == [(10, 20), (10, 30), (20, 30)]
    assert torch.isclose(value, torch.tensor((200.0 + 300.0 + 600.0) / 3.0))


def test_batch_correction_mmd_is_invariant_to_species_imbalance_at_aggregation_level():
    _, balanced_mmd = _load_mmd_helpers()

    def fake_mmd(left, right):
        return left[:, 0].mean() * right[:, 0].mean()

    balanced_mmd.__globals__["mmd_loss"] = fake_mmd
    balanced_embeddings = torch.tensor([[10.0], [20.0], [30.0]])
    balanced_species = torch.tensor([1, 2, 3])
    imbalanced_embeddings = torch.tensor(
        [[10.0], [10.0], [10.0], [10.0], [20.0], [30.0], [30.0]]
    )
    imbalanced_species = torch.tensor([1, 1, 1, 1, 2, 3, 3])

    balanced_value = balanced_mmd(balanced_embeddings, balanced_species)
    imbalanced_value = balanced_mmd(imbalanced_embeddings, imbalanced_species)

    assert torch.isclose(imbalanced_value, balanced_value)


def test_batch_correction_mmd_does_not_depend_on_cell_type_labels():
    source = ast.get_source_segment(_source(), _function_node("batch_corr_pass"))

    assert "self.predict_keys.index(self.do_mmd_on)" not in source
    assert "class_elem[:, class_pos]" not in source


def test_balanced_species_pairwise_mmd_keeps_gradients():
    _, balanced_mmd = _load_mmd_helpers()
    embeddings = torch.tensor([[0.0], [3.0], [7.0]], requires_grad=True)
    species = torch.tensor([1, 2, 3])

    value = balanced_mmd(embeddings, species)

    assert value.requires_grad
    value.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_legacy_mmd_path_returns_a_detached_python_scalar():
    legacy_mmd = _load_legacy_mmd_helpers()
    embeddings = torch.tensor([[0.0], [1.0], [4.0], [6.0]], requires_grad=True)
    species = torch.tensor([0, 0, 1, 1])

    value = legacy_mmd(embeddings, species)

    assert isinstance(value, float)
    assert embeddings.grad is None


def test_energy_mmd_supports_singletons_and_gradients():
    mmd_loss = _load_mmd_loss()
    x = torch.tensor([[0.0, 0.0]], requires_grad=True)
    y = torch.tensor([[3.0, 4.0]], requires_grad=True)

    value = mmd_loss(x, y)

    assert torch.isclose(value, torch.tensor(10.0))
    value.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()
    assert x.grad.abs().sum() > 0
    assert y.grad.abs().sum() > 0


def test_energy_mmd_matches_notebook_unbiased_estimator():
    mmd_loss = _load_mmd_loss()
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    y = x.clone().detach().requires_grad_(True)
    identical = mmd_loss(x, y)
    pair_distance = torch.dist(x[0], x[1])

    # The notebook uses the unbiased within-group terms, so an identical
    # finite sample can have a negative estimate instead of being clamped.
    assert torch.isclose(identical, -pair_distance)


def test_mmd_requires_two_groups_before_training():
    source = _source()
    call_source = ast.get_source_segment(source, _function_node("__call__"))

    assert "MMD requires at least two values" in call_source


def test_cpu_optimizer_step_does_not_require_grad_scaler():
    tree = ast.parse(_source())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_optimizer_step"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_value = model(torch.ones((1, 1))).sum()
    namespace["_optimizer_step"](loss_value, optimizer, model, scaler=None)

    assert parameter.item() < 1.0


def test_trainable_state_restore_only_restores_trainable_parameters():
    trainable_state_dict, restore_trainable_state = _load_trainable_state_helpers()

    module = torch.nn.Module()
    module.trainable = torch.nn.Parameter(torch.tensor([1.0]))
    module.frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    state = trainable_state_dict(module)
    with torch.no_grad():
        module.trainable.fill_(10.0)
        module.frozen.fill_(20.0)
    restore_trainable_state(module, state)

    assert torch.equal(module.trainable, torch.tensor([1.0]))
    assert torch.equal(module.frozen, torch.tensor([20.0]))
