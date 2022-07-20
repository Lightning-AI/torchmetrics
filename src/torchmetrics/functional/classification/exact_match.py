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
from typing import Optional, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.stat_scores import (
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
)
from torchmetrics.utilities.data import _movedim


def _multilabel_exact_scores_update(
    preds: Tensor, target: Tensor, num_labels: int, multidim_average: Literal["global", "samplewise"] = "global"
) -> Tuple[Tensor, int]:
    """Computes the statistics."""
    if multidim_average == "global":
        preds = _movedim(preds, 1, -1).reshape(-1, num_labels)
        target = _movedim(target, 1, -1).reshape(-1, num_labels)

    correct = ((preds == target).sum(1) == num_labels).sum(dim=-1)
    total = (target != -1).sum()  # -1 indicates it should be ignored
    total = preds.shape[0 if multidim_average == "global" else 2]
    return correct, total


def _multilabel_stat_scores_compute(
    correct: Tensor,
    total: Union[int, Tensor],
) -> Tensor:
    return correct / total


def multilabel_exact_match(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    average = None
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    correct, total = _multilabel_exact_scores_update(preds, target, num_labels, multidim_average)
    return _multilabel_stat_scores_compute(correct, total)
