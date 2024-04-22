# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.functional.segmentation.utils import _ignore_background
from typing_extensions import Literal

def _generalized_dice_validate_args(
    num_classes: int,
    include_background: bool,
    per_class: bool,
) -> None:
    """Validate the arguments of the metric."""
    if num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    if not isinstance(per_class, bool):
        raise ValueError(f"Expected argument `per_class` must be a boolean, but got {per_class}.")
    

def _generalized_dice_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    per_class: bool,
    weight_type: Literal["square", "simple", "linear"] = "square",
) -> Tensor:
    """Update the state with the current prediction and target."""
    _check_same_shape(preds, target)
    if preds.ndim < 3:
        
    
    if (preds.bool() != preds).any():  # preds is an index tensor
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
    if (target.bool() != target).any():  # target is an index tensor
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if not include_background:
        preds, target = _ignore_background(preds, target)

    reduce_axis = list(range(2, target.ndim))
    intersection = torch.sum(preds * target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    cardinality = target_sum + pred_sum

    if weight_type == "simple":
        weights = 1.0 / target_sum
    elif weight_type == "linear":
        weights = torch.ones_like(target_sum)
    elif weight_type == "square":
        weights = 1.0 / (target_sum ** 2)
    else:
        raise ValueError(
            f"Expected argument `weight_type` to be one of 'simple', 'linear', 'square', but got {weight_type}."
        )
    
    infs = torch.isinf(weights)
    weights[infs] = 0
    weights = torch.max(weights, 0)

    numerator = 2.0 * (intersection * weights).sum(dim=1)
    denominator = (cardinality * weights).sum(dim=1)
    return numerator, denominator

def _generalized_dice_compute(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Compute the generalized dice score."""
    return _safe_divide(numerator, denominator)


def generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    per_class: bool = False,
) -> Tensor:
    """
    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.functional.segmentation import generalized_dice_score
        >>> preds = torch.randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = torch.randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
    """
    _generalized_dice_validate_args(num_classes, include_background, per_class)
    numerator, denominator = _generalized_dice_update(preds, target, num_classes, include_background, per_class)
    return _generalized_dice_compute(numerator, denominator)
