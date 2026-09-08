"""Compatibility wrapper for OpenProblems-compatible scIB evaluation helpers."""

from scprint2.evaluation import openproblems_scib as _impl
from scprint2.evaluation.openproblems_scib import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_impl)})
