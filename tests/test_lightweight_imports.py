from __future__ import annotations

import subprocess
import sys


def test_task3_helpers_import_without_model_training_dependencies():
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'lightning' or name.startswith('lightning.'):
        raise RuntimeError('task3 helper import eagerly imported lightning')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from scprint2.benchmark.task3 import Task3Config
print(Task3Config.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Task3Config"
