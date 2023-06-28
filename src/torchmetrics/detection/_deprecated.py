from typing import Any, Collection

from torchmetrics.detection import ModifiedPanopticQuality, PanopticQuality
from torchmetrics.utilities.prints import _deprecated_root_import_class


class _ModifiedPanopticQuality(ModifiedPanopticQuality):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([[[0, 0], [0, 1], [6, 0], [7, 0], [0, 2], [1, 0]]])
    >>> target = tensor([[[0, 1], [0, 0], [6, 0], [7, 0], [6, 0], [255, 0]]])
    >>> pq_modified = _ModifiedPanopticQuality(things = {0, 1}, stuffs = {6, 7})
    >>> pq_modified(preds, target)
    tensor(0.7667, dtype=torch.float64)
    """

    def __init__(
        self,
        things: Collection[int],
        stuffs: Collection[int],
        allow_unknown_preds_category: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("ModifiedPanopticQuality", "detection")
        super().__init__(
            things=things, stuffs=stuffs, allow_unknown_preds_category=allow_unknown_preds_category, **kwargs
        )


class _PanopticQuality(PanopticQuality):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([[[[6, 0], [0, 0], [6, 0], [6, 0]],
    ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
    ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
    ...                  [[0, 0], [7, 0], [6, 0], [1, 0]],
    ...                  [[0, 0], [7, 0], [7, 0], [7, 0]]]])
    >>> target = tensor([[[[6, 0], [0, 1], [6, 0], [0, 1]],
    ...                   [[0, 1], [0, 1], [6, 0], [0, 1]],
    ...                   [[0, 1], [0, 1], [6, 0], [1, 0]],
    ...                   [[0, 1], [7, 0], [1, 0], [1, 0]],
    ...                   [[0, 1], [7, 0], [7, 0], [7, 0]]]])
    >>> panoptic_quality = _PanopticQuality(things = {0, 1}, stuffs = {6, 7})
    >>> panoptic_quality(preds, target)
    tensor(0.5463, dtype=torch.float64)
    """

    def __init__(
        self,
        things: Collection[int],
        stuffs: Collection[int],
        allow_unknown_preds_category: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("PanopticQuality", "detection")
        super().__init__(
            things=things, stuffs=stuffs, allow_unknown_preds_category=allow_unknown_preds_category, **kwargs
        )
