"""Benchmark runner utilities."""

from importlib import import_module

__all__ = [
    "OpenProblemsConfig",
    "OpenProblemsResult",
    "OpenProblemsV1Config",
    "OpenProblemsV1Result",
    "Task3Config",
    "Task3RunArtifacts",
]

_EXPORT_MODULES = {
    "OpenProblemsConfig": ".openproblems",
    "OpenProblemsResult": ".openproblems",
    "OpenProblemsV1Config": ".openproblems_v1",
    "OpenProblemsV1Result": ".openproblems_v1",
    "Task3Config": ".task3",
    "Task3RunArtifacts": ".task3",
}


def __getattr__(name: str):
    """Import only the requested benchmark surface and its dependencies."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name, __name__), name)
