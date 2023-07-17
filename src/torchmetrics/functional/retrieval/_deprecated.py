from typing import Optional, Tuple

from torch import Tensor

from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.functional.retrieval.r_precision import retrieval_r_precision
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from torchmetrics.utilities.prints import _deprecated_root_import_func


def _retrieval_average_precision(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_average_precision(preds, target)
    tensor(0.8333)
    """
    _deprecated_root_import_func("retrieval_average_precision", "retrieval")
    return retrieval_average_precision(preds=preds, target=target, top_k=top_k)


def _retrieval_fall_out(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_fall_out(preds, target, top_k=2)
    tensor(1.)
    """
    _deprecated_root_import_func("retrieval_fall_out", "retrieval")
    return retrieval_fall_out(preds=preds, target=target, top_k=top_k)


def _retrieval_hit_rate(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_hit_rate(preds, target, top_k=2)
    tensor(1.)
    """
    _deprecated_root_import_func("retrieval_hit_rate", "retrieval")
    return retrieval_hit_rate(preds=preds, target=target, top_k=top_k)


def _retrieval_normalized_dcg(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([.1, .2, .3, 4, 70])
    >>> target = tensor([10, 0, 0, 1, 5])
    >>> _retrieval_normalized_dcg(preds, target)
    tensor(0.6957)
    """
    _deprecated_root_import_func("retrieval_normalized_dcg", "retrieval")
    return retrieval_normalized_dcg(preds=preds, target=target, top_k=top_k)


def _retrieval_precision(
    preds: Tensor, target: Tensor, top_k: Optional[int] = None, adaptive_k: bool = False
) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_precision(preds, target, top_k=2)
    tensor(0.5000)
    """
    _deprecated_root_import_func("retrieval_precision", "retrieval")
    return retrieval_precision(preds=preds, target=target, top_k=top_k, adaptive_k=adaptive_k)


def _retrieval_precision_recall_curve(
    preds: Tensor, target: Tensor, max_k: Optional[int] = None, adaptive_k: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> precisions, recalls, top_k = _retrieval_precision_recall_curve(preds, target, max_k=2)
    >>> precisions
    tensor([1.0000, 0.5000])
    >>> recalls
    tensor([0.5000, 0.5000])
    >>> top_k
    tensor([1, 2])
    """
    _deprecated_root_import_func("retrieval_precision_recall_curve", "retrieval")
    return retrieval_precision_recall_curve(preds=preds, target=target, max_k=max_k, adaptive_k=adaptive_k)


def _retrieval_r_precision(preds: Tensor, target: Tensor) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_r_precision(preds, target)
    tensor(0.5000)
    """
    _deprecated_root_import_func("retrieval_r_precision", "retrieval")
    return retrieval_r_precision(preds=preds, target=target)


def _retrieval_recall(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_recall(preds, target, top_k=2)
    tensor(0.5000)
    """
    _deprecated_root_import_func("retrieval_recall", "retrieval")
    return retrieval_recall(preds=preds, target=target, top_k=top_k)


def _retrieval_reciprocal_rank(preds: Tensor, target: Tensor) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([False, True, False])
    >>> _retrieval_reciprocal_rank(preds, target)
    tensor(0.5000)
    """
    _deprecated_root_import_func("retrieval_reciprocal_rank", "retrieval")
    return retrieval_reciprocal_rank(preds=preds, target=target)
