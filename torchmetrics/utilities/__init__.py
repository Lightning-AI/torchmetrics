from typing import Any, Optional
from warnings import warn

from torchmetrics.utilities.data import apply_to_collection  # noqa: F401
from torchmetrics.utilities.distributed import class_reduce, reduce  # noqa: F401
from torchmetrics.utilities.prints import rank_zero_debug, rank_zero_info, rank_zero_warn  # noqa: F401


def _deprecation_warn_arg_multilabel(arg: Any) -> None:
    if arg is None:
        return
    warn(
        "Argument `multilabel` was deprecated in v0.3 and will be removed in v0.4. Use `multiclass` instead.",
        DeprecationWarning
    )


def _deprecation_warn_arg_is_multiclass(arg_old: Optional[bool], arg_new: Optional[bool]) -> bool:
    if arg_old is None:
        return arg_new
    warn(
        "Argument `is_multiclass` was deprecated in v0.3 and will be removed in v0.4. Use `multiclass` instead.",
        DeprecationWarning
    )
    if arg_new is None:
        arg_new = arg_old
    return arg_new
