from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scprint2")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["scPRINT2"]


def __getattr__(name: str):
    """Load the model only when requested so lightweight helpers stay importable."""
    if name == "scPRINT2":
        from .model.model import scPRINT2

        return scPRINT2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# from .data_collator import DataCollator
# from .data_sampler import SubsetsBatchSampler
# from .trainer import (
#    prepare_data,
#    prepare_dataloader,
#    train,
#    define_wandb_metrcis,
#    evaluate,
#    eval_testdata,
#    test,
# )#
