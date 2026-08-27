from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_adoption_module():
    path = ROOT / "scripts" / "training" / "adopt_primary_filtered_cache.py"
    spec = importlib.util.spec_from_file_location("adopt_primary_filtered_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_partial_cache_train_weights_match_current_sampler_formula():
    module = load_adoption_module()
    labels = np.array([0, 0, 1, 2, 2, 3])
    result = module.derive_train_weights(labels, weight_scaler=10)
    expected = np.array([20 / 12, 10 / 11, 20 / 12, 0], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_launchers_are_current_main_immutable_and_do_not_rebuild_cache():
    for name in ("scprint2_r3_primary_smoke.sbatch", "scprint2_r3_primary_long.sbatch"):
        source = (ROOT / "slurm" / name).read_text()
        assert "config/base_v3_ntv3_primary.yml" in source
        assert '"$ENV/bin/scprint2" fit' in source
        assert "EXPECTED_SCPRINT_COMMIT" in source
        assert "0a2fb2080dd2bf471c1532c04ee0789ce06441c2" in source
        assert "prepare_primary_filtered_cache.py" not in source
        assert "control_primary_v" not in source
    smoke = (ROOT / "slurm" / "scprint2_r3_primary_smoke.sbatch").read_text()
    assert "adopt_primary_filtered_cache.py" in smoke
    assert "GPUTelemetryGuard" in smoke
    assert "fallback/gpu_count" in smoke
    assert "WANDB_DIR=$TMPBASE/wandb" in smoke
    long = (ROOT / "slurm" / "scprint2_r3_primary_long.sbatch").read_text()
    assert "WANDB_DIR=$TMPBASE/wandb" in long
