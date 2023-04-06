from typing import Collection

from torch import Tensor

from torchmetrics.functional.detection.panoptic_qualities import modified_panoptic_quality, panoptic_quality
from torchmetrics.utilities.prints import _deprecated_root_import_func


def _modified_panoptic_quality(
    preds: Tensor,
    target: Tensor,
    things: Collection[int],
    stuffs: Collection[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([[[0, 0], [0, 1], [6, 0], [7, 0], [0, 2], [1, 0]]])
    >>> target = tensor([[[0, 1], [0, 0], [6, 0], [7, 0], [6, 0], [255, 0]]])
    >>> _modified_panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
    tensor(0.7667, dtype=torch.float64)
    """
    _deprecated_root_import_func("modified_panoptic_quality", "detection")
    return modified_panoptic_quality(
        preds=preds,
        target=target,
        things=things,
        stuffs=stuffs,
        allow_unknown_preds_category=allow_unknown_preds_category,
    )


def _panoptic_quality(
    preds: Tensor,
    target: Tensor,
    things: Collection[int],
    stuffs: Collection[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
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
    >>> _panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
    tensor(0.5463, dtype=torch.float64)
    """
    _deprecated_root_import_func("panoptic_quality", "detection")
    return panoptic_quality(
        preds=preds,
        target=target,
        things=things,
        stuffs=stuffs,
        allow_unknown_preds_category=allow_unknown_preds_category,
    )
