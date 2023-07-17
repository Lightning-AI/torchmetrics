from typing import Any, Optional

from torchmetrics.retrieval.average_precision import RetrievalMAP
from torchmetrics.retrieval.fall_out import RetrievalFallOut
from torchmetrics.retrieval.hit_rate import RetrievalHitRate
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from torchmetrics.retrieval.precision import RetrievalPrecision
from torchmetrics.retrieval.precision_recall_curve import RetrievalPrecisionRecallCurve, RetrievalRecallAtFixedPrecision
from torchmetrics.retrieval.r_precision import RetrievalRPrecision
from torchmetrics.retrieval.recall import RetrievalRecall
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR
from torchmetrics.utilities.prints import _deprecated_root_import_class


class _RetrievalFallOut(RetrievalFallOut):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> fo = _RetrievalFallOut(top_k=2)
    >>> fo(preds, target, indexes=indexes)
    tensor(0.5000)
    """

    def __init__(
        self,
        empty_target_action: str = "pos",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalFallOut", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)


class _RetrievalHitRate(RetrievalHitRate):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([True, False, False, False, True, False, True])
    >>> hr2 = _RetrievalHitRate(top_k=2)
    >>> hr2(preds, target, indexes=indexes)
    tensor(0.5000)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalHitRate", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)


class _RetrievalMAP(RetrievalMAP):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> rmap = _RetrievalMAP()
    >>> rmap(preds, target, indexes=indexes)
    tensor(0.7917)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalMAP", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)


class _RetrievalRecall(RetrievalRecall):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> r2 = _RetrievalRecall(top_k=2)
    >>> r2(preds, target, indexes=indexes)
    tensor(0.7500)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalRecall", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)


class _RetrievalRPrecision(RetrievalRPrecision):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> p2 = _RetrievalRPrecision()
    >>> p2(preds, target, indexes=indexes)
    tensor(0.7500)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalRPrecision", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, **kwargs)


class _RetrievalNormalizedDCG(RetrievalNormalizedDCG):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> ndcg = _RetrievalNormalizedDCG()
    >>> ndcg(preds, target, indexes=indexes)
    tensor(0.8467)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalNormalizedDCG", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)


class _RetrievalPrecision(RetrievalPrecision):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> p2 = _RetrievalPrecision(top_k=2)
    >>> p2(preds, target, indexes=indexes)
    tensor(0.5000)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        adaptive_k: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("", "retrieval")
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            top_k=top_k,
            adaptive_k=adaptive_k,
            **kwargs,
        )


class _RetrievalPrecisionRecallCurve(RetrievalPrecisionRecallCurve):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 0, 1, 1, 1])
    >>> preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
    >>> target = tensor([True, False, False, True, True, False, True])
    >>> r = _RetrievalPrecisionRecallCurve(max_k=4)
    >>> precisions, recalls, top_k = r(preds, target, indexes=indexes)
    >>> precisions
    tensor([1.0000, 0.5000, 0.6667, 0.5000])
    >>> recalls
    tensor([0.5000, 0.5000, 1.0000, 1.0000])
    >>> top_k
    tensor([1, 2, 3, 4])
    """

    def __init__(
        self,
        max_k: Optional[int] = None,
        adaptive_k: bool = False,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("", "retrieval")
        super().__init__(
            max_k=max_k,
            adaptive_k=adaptive_k,
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )


class _RetrievalRecallAtFixedPrecision(RetrievalRecallAtFixedPrecision):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 0, 1, 1, 1])
    >>> preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
    >>> target = tensor([True, False, False, True, True, False, True])
    >>> r = _RetrievalRecallAtFixedPrecision(min_precision=0.8)
    >>> r(preds, target, indexes=indexes)
    (tensor(0.5000), tensor(1))
    """

    def __init__(
        self,
        min_precision: float = 0.0,
        max_k: Optional[int] = None,
        adaptive_k: bool = False,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("RetrievalRecallAtFixedPrecision", "retrieval")
        super().__init__(
            min_precision=min_precision,
            max_k=max_k,
            adaptive_k=adaptive_k,
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )


class _RetrievalMRR(RetrievalMRR):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([False, False, True, False, True, False, True])
    >>> mrr = _RetrievalMRR()
    >>> mrr(preds, target, indexes=indexes)
    tensor(0.7500)
    """

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("", "retrieval")
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, **kwargs)
